use oh_my_rust::*;
use crate::misc::*;
use crate::graph::*;
use crate::proto::graph::GraphDef;
use crate::proto::node_def::NodeDef;

// if we do not remove these, we need to modify this field so that it has the correct node name of replicated operators
pub fn remove_collocation_hint(target: &mut Target) {
    for node in target.pb.node.iter_mut() {
        if let Some(x) = node.attr.get_mut("_class") {
            if let Some(crate::proto::attr_value::AttrValue_oneof_value::list(ref mut list)) = &mut x.value {
                list.s = list.s.iter().filter(|x| !x.starts_with(b"loc:")).cloned().collect()
            }
        }
    }
}

pub fn remove_shape_hint(target: &mut Target) {
    for node in target.pb.node.iter_mut() {
        node.attr.remove("_output_shapes");
    }
}

pub fn remove_dangling_nodes(target: &mut Target) {
    // note: don't forget control dependency
    let input_node_dict: std::collections::HashMap<_, Vec<_>> = target.pb.node.iter().map(|node| {
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
            queue.extend(&input_node_dict[x]);
        }
    }

    // hacky way to avoid clone
    let mut x = std::mem::replace(&mut target.pb.node, vec![].into()).into_vec();
    x.retain(|x| keep.contains(&x.name[..]));
    target.pb.node = x.into()
}

pub fn destruct_names(target: &mut Target) {
    for node in target.pb.node.iter_mut() {
        node.name = node.name.replace('/', "__");
        for input in node.input.iter_mut() {
            *input = input.replace('/', "__");
        }
    }
}

pub fn fuse_mini_batch(nodes: &[NodeDef], times: usize) -> Vec<NodeDef> {
    let mut result = Vec::with_capacity(nodes.len() * times);

    for i in 0..times {
        for mut node in nodes.iter().cloned() {
            node.name = format!("tge_fuse_batch_{}/{}", i, node.name);
            for input in node.input.iter_mut() {
                if input.starts_with('^') {
                    *input = format!("^tge_fuse_batch_{}/{}", i, &input[1..]);
                } else {
                    *input = format!("tge_fuse_batch_{}/{}", i, input);
                }
            }
            result.push(node)
        }
    }

    todo!()
}

// remove identity nodes, NoOp nodes, and control dependencies, except for sinks
pub fn merge_trivial_nodes(target: &mut Target) {
    let mut merged: std::collections::BTreeMap<String, String> = Default::default();
    target.pb.node = core::mem::take(&mut target.pb.node).into_iter().filter(|node| {
        if target.sinks.contains(&node.name) {
            return true
        }

        match &node.op[..] {
            "NoOp" => {
                return false
            }
            "Identity" | "Sigmoid" | "LeakyRelu" | "Relu" | "Tanh" => { // TODO: save the name of merged node into attr of the parent and look up for the computation time when simulating
                merged.insert(node.name.clone(), node.input[0].clone());
                return false
            }
            _ => return true
        }
    }).collect();

    for node in target.pb.node.iter_mut() {
        if target.sinks.contains(&node.name) { // what if the sink node refers to a node that is merged?
            continue
        }

        node.input = core::mem::take(&mut node.input).into_iter().filter_map(|input| {
            if input.starts_with('^') {
                return None
            }

            if let Some(new_name) = merged.get(&input) { // by default Tensorflow omit :0 for the first output
                return Some(new_name.clone())
            } else {
                return Some(input)
            }
        }).collect();
    }
}

