use oh_my_rust::*;
use core::convert::TryInto;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque, HashMap};
use std::sync::{Arc, Mutex};
use core::cmp;
use crate::misc::{Target, Profiler};
use crate::graph::Form;
use crate::proto::types::DataType;
use crate::proto::attr_value::{AttrValue, AttrValue_oneof_value};
use crate::proto::node_def::NodeDef;
use crate::proto::tensor::TensorProto;
use crate::simulator::{GRPC_LATENCY, FALLBACK_NCCL_MODEL};

pub fn heft_control(target: &mut Target, profiler: &impl Profiler) {
    heft_rank(target, profiler, true);

    let name_dict: BTreeMap<String, usize> = target.pb.node.iter().enumerate().map(|(i, x)| (x.name.clone(), i)).collect();
    let non_dangling_nodes = mark_non_dangling_nodes(target);

    for dev in target.devices.iter() {
        let mut list: Vec<_> = (0..name_dict.len()).filter(|&i| {
            let node = &target.pb.node[i];
            non_dangling_nodes.contains(&node.name) && *dev == node.device
        }).collect();

        list.sort_unstable_by_key(|&i| target.pb.node[i].attr.get("_priority").unwrap().get_i());
        for window in list.windows(2) {
            let dep = format!("^{}", target.pb.node[window[0]].name);
            target.pb.node[window[1]].input.push(dep)
            // TODO: skip unnecessary dependencies by depth-first search?
        }
    }
}

pub fn heft_rank(target: &mut Target, profiler: &impl Profiler, break_tie: bool) {
    let name_dict: BTreeMap<String, usize> = target.pb.node.iter().enumerate().map(|(i, x)| (x.name.clone(), i)).collect();
    let device_dict: BTreeMap<&String, usize> = target.devices.iter().enumerate().map(|(i, x)| (x, i)).collect();
    let mut ranks = vec![Option::<u64>::None; name_dict.len()];
    let mut succs = vec![BTreeSet::<usize>::new(); name_dict.len()];

    for (node, succ) in target.pb.node.iter().zip(succs.iter_mut()) {
        for input in node.input.iter() {
            let input = if input.starts_with('^') {
                &input[1..]
            } else {
                parse_input(input).0
            };

            succ.insert(name_dict[input]);
        }
    }

    let mut stack: Vec<_> = (0..name_dict.len()).collect();
    while let Some(i) = stack.pop() {
        if ranks[i].is_some() {
            continue
        }

        if succs[i].iter().any(|&j| ranks[j].is_none()) {
            stack.push(i);
            stack.extend(succs[i].iter());
            continue
        }

        let device_id = device_dict[&target.pb.node[i].device];
        let time = succs[i].iter().map(|&j| ranks[j].unwrap()).max().unwrap_or(0) +
                   profiler.profile(&target.pb.node[i], device_id).unwrap_or(0) +
                   break_tie as u64; // additional rank to prevent ties on zero-time op which may cause dead locks
        ranks[i] = Some(time)
    }

    let non_dangling_nodes = mark_non_dangling_nodes(target);
    for (node, rank) in target.pb.node.iter_mut().zip(ranks) {
        if non_dangling_nodes.contains(&node.name) {
            node.attr.insert("_priority".to_string(), AttrValue::new().apply(|x| x.set_i(rank.unwrap() as _))).ignore();
        }
    }
}

fn parse_input(x: &str) -> (&str, usize) {
    match x.find(':') {
        Some(i) => (&x[..i], x[i+1..].parse().unwrap()),
        None => (x, 0)
    }
}

pub fn mark_non_dangling_nodes(target: &Target) -> std::collections::HashSet<String> {
    // note: don't forget control dependency
    let dict: std::collections::HashMap<_, Vec<_>> = target.pb.node.iter().map(|node| {
        (&node.name[..], node.input.iter().map(|x| {
            if x.starts_with('^') {
                return &x[1..]
            }
            match x.find(':') {
                Some(i) => &x[..i],
                None => x
            }
        }).collect())
    }).collect();
    let mut keep = std::collections::HashSet::new();
    let mut queue: std::collections::VecDeque<_> = target.sinks.iter().map(|x| &x[..]).collect();

    while let Some(x) = queue.pop_front() {
        if keep.insert(x.to_string()) {
            queue.extend(&dict[x]);
        }
    }

    keep
}
