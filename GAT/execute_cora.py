import time
import numpy as np
import tensorflow as tf

import google.protobuf.text_format as pbtf
from tensorflow.core.framework import graph_pb2
from sklearn.preprocessing import StandardScaler
import copy
import sys
import os
import scipy.sparse as sp
import traceback
import pickle
sys.path.append('../')
import tge
import json
import pickle as pkl
import multiprocessing as mp

from utils import group_around_topk_costs
import logging
import math
import dgl
from tf2gat import GAT

def InitLog():
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    # log to txt
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.FileHandler("log/log_%s.txt" % time.strftime("%Y-%m-%d-%H-%M-%S"))
    # handler = logging.handlers.RotatingFileHandler("log_%s.txt" % time.strftime("%Y-%m-%d %H-%M-%S"),maxBytes=1024*1024,backupCount=50)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    # log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    log.addHandler(handler)
    log.addHandler(console)
    return log

logger = InitLog()
variable_ops=["Variable", "VariableV2", "AutoReloadVariable",
                   "MutableHashTable", "MutableHashTableV2",
                   "MutableHashTableOfTensors", "MutableHashTableOfTensorsV2",
                   "MutableDenseHashTable", "MutableDenseHashTableV2",
                   "VarHandleOp", "BoostedTreesEnsembleResourceHandleOp",
                   "BoostedTreesQuantileStreamResourceHandleOp",
                   "ResourceConditionalAccumulator",
                   "DecisionTreeResource"]


checkpt_file = 'pre_trained/cora/mod_cora.ckpt'
_dataset = 'cora'

config_dict = dict()
if os.path.exists("config.txt"):
    with open("config.txt", "r") as f:
        config_dict = json.load(f)

# training params
os.environ["CUDA_VISIBLE_DEVICES"]=config_dict.get("CUDA_VISIBLE_DEVICES","0,1")
#os.environ["TF_XLA_FLAGS"]="--tf_xla_cpu_global_jit"
batch_size = 1
nb_epochs = 100000
patience = 100
lr = config_dict.get("learning_rate", 0.01)  # learning rate
l2_coef = 0.0002  # weight decay
hid_units = [512]  # numbers of hidden units per each attention head in each layer
n_heads = [4, 4]  # additional entry for the output layer
place_hid_units = [1024, 256]
place_n_heads = [4,4,1]
residual = False


global_batch_size=288

n_layer=12
n_head=8
d_head=64
d_model=512
d_inner=2048
group_num = 2000
bsz =1




nonlinearity = tf.nn.elu
is_transformer = True
print('Dataset: ' + _dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
feature_folders = config_dict.get("inputs",["data/graph1", "data/graph2", "data/graph3", "data/graph4", "data/graph5", "data/graph6","data/graph7","data/graph8"])
sinks =  config_dict.get("sinks",[["GradientDescent"], ["GradientDescent"], ["GradientDescent"], ["GradientDescent"], ["GradientDescent"], ["GradientDescent"],["GradientDescent"],["GradientDescent"]])
sample_times = 3
devices = config_dict.get("devices", [
    "/job:worker/replica:0/task:0/device:GPU:0",
    "/job:worker/replica:0/task:0/device:GPU:1",
    "/job:worker/replica:0/task:1/device:GPU:0",
    "/job:worker/replica:0/task:1/device:GPU:1",
    "/job:worker/replica:0/task:2/device:GPU:0",
    "/job:worker/replica:0/task:2/device:GPU:1"

])

max_replica_num = config_dict.get("max_replica_num", len(devices))
show_interval = 3
num_cores = mp.cpu_count()
device_mems = config_dict.get("device_mems", [16 * 10e9, 16 * 10e9, 16 * 10e9, 16 * 10e9])

sample_prob = 0.1


def post_process_device_choice(device_choice,batch_size):
    def post_func1(item):
        item1 = list(item[:len(item) - 1])
        batch_size = item[-1]
        if sum(item1) == 0:
            item1[0] = 1
            return item1
        while sum(item1) > batch_size:
            index = item1.index(max(item1))
            item1[index] -= 1
        while batch_size % sum(item1):
            index = item1.index(max(item1))
            item1[index] -= 1
        return np.array(item1)
    #post process and align to batch size
    new_batch_size=np.ones(shape=(device_choice.shape[0],1)) * batch_size
    device_choice=np.array(list(map(post_func1,np.concatenate((device_choice,new_batch_size),axis=1))),dtype=np.int32)

    replica_mask = np.zeros(shape=(device_choice.shape[0],device_choice.shape[1]*(max_replica_num+1)+2),dtype=np.int32)
    for i,item in enumerate(device_choice):
        for j,num in enumerate(item):
            replica_mask[i][j*(max_replica_num+1)+num]=1
    return device_choice,replica_mask


class strategy_pool(object):
    def __init__(self,folder_path,node_num,env,batch_size,init_group,sink):
        self.folder_path = folder_path
        self.node_num = node_num
        self.env = env
        self.sink = sink
        self.init_group = init_group
        self.init_group_num = max(self.init_group)+1
        if os.path.exists(self.folder_path+"/pool.pkl"):
            with open(self.folder_path+"/pool.pkl","rb") as f:
                self.strategies= mp.Manager().list(pkl.load(f))
                for j, strategy in enumerate(self.strategies):
                    group = strategy["group"]
                    if len(group)!=self.init_group_num:
                        self.strategies.pop(j)
            self.save_strategy_pool()
        else:
            self.strategies = mp.Manager().list()

        self.rewards = [item["reward"] for item in self.strategies] if len(self.strategies) else [-sys.maxsize]
        self.batch_size = batch_size

        # even data parallel 1
        #device_choice = np.zeros(shape=(self.node_num, len(devices)), dtype=np.int32)
        if False:
            group = np.array(self.init_group)
            device_choice = np.ones(shape=(self.init_group_num, len(devices)), dtype=np.int32)*2

            device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
            ps_or_reduce = np.ones(shape=(self.init_group_num, ), dtype=np.int32)
            reward,out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.sink,group,record=True,record_name="full_nccl_dp_graph.pbtxt",record_best=False,from_strategy_pool=True)
            if not out_of_memory:
                self.insert(reward, device_choice, replica_mask,ps_or_reduce,group,force_insert=True)


            group = np.array(self.init_group)
            device_choice = np.ones(shape=(self.init_group_num, len(devices)), dtype=np.int32)
            for item in device_choice:
                item[0]=2
                item[1]=2
            device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
            ps_or_reduce = np.ones(shape=(self.init_group_num, ), dtype=np.int32)
            reward,out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce,self.sink,group,record=True,record_name="partial_nccl_dp_graph.pbtxt",record_best=False,from_strategy_pool=True)
            if not out_of_memory:
                self.insert(reward, device_choice, replica_mask,ps_or_reduce,group,force_insert=True)


            # even data parallel 2
            #device_choice = np.zeros(shape=(self.node_num, len(devices)), dtype=np.int32)

            group = np.array(self.init_group)
            device_choice = np.ones(shape=(self.init_group_num, len(devices)), dtype=np.int32)

            device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
            ps_or_reduce = np.ones(shape=(self.init_group_num, ), dtype=np.int32)
            reward,out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce,self.sink,group,record=True,record_name="nccl_dp_graph.pbtxt",record_best=False,from_strategy_pool=True)
            if not out_of_memory:
                self.insert(reward, device_choice, replica_mask,ps_or_reduce,group,force_insert=True)

        #    self.insert(reward, device_choice, replica_mask,ps_or_reduce,group)

            group = np.array(self.init_group)
            device_choice = np.ones(shape=(self.init_group_num, len(devices)), dtype=np.int32)

            device_choice, replica_mask = post_process_device_choice(device_choice, self.batch_size)
            ps_or_reduce = np.zeros(shape=(self.init_group_num,), dtype=np.int32)
            reward, out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.sink, group,
                                                         record=True, record_name="grpc_dp_graph.pbtxt", record_best=False,
                                                         from_strategy_pool=True)
            if not out_of_memory:
                self.insert(reward, device_choice, replica_mask,ps_or_reduce,group,force_insert=True)

            #    self.insert(reward, device_choice, replica_mask,ps_or_reduce,group)


            group = np.array(self.init_group)
            device_choice = np.array([np.arange(len(devices))%1 for i in range(self.init_group_num)])

            device_choice, replica_mask = post_process_device_choice(device_choice, self.batch_size)
            ps_or_reduce = np.ones(shape=(self.init_group_num,), dtype=np.int32)
            reward, out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.sink, group,
                                                         record=True, record_name="single_graph.pbtxt",
                                                         record_best=False, from_strategy_pool=True)
            if not out_of_memory:
                self.insert(reward, device_choice, replica_mask,ps_or_reduce,group,force_insert=True)

            group = np.array(self.init_group)
            device_choice = np.array([np.arange(len(devices))%1 for i in range(self.init_group_num)])
            for i,item in enumerate(device_choice):
                item[i%len(devices)]=1

            device_choice, replica_mask = post_process_device_choice(device_choice, self.batch_size)
            ps_or_reduce = np.ones(shape=(self.init_group_num,), dtype=np.int32)
            reward, out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.sink, group,
                                                         record=True, record_name="model_parallel_graph.pbtxt",
                                                         record_best=False, from_strategy_pool=True)
            if not out_of_memory:
                self.insert(reward, device_choice, replica_mask,ps_or_reduce,group,force_insert=True)


            group = np.array(self.init_group)
            device_choice = np.array([np.arange(len(devices))%1 for i in range(self.init_group_num)])
            for i,item in enumerate(device_choice):
                item[i//math.ceil(len(device_choice)/len(devices))]=1

            device_choice, replica_mask = post_process_device_choice(device_choice, self.batch_size)
            ps_or_reduce = np.ones(shape=(self.init_group_num,), dtype=np.int32)
            reward, out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.sink, group,
                                                         record=True, record_name="model_parallel2_graph.pbtxt",
                                                         record_best=False, from_strategy_pool=True)
            if not out_of_memory:
                self.insert(reward, device_choice, replica_mask,ps_or_reduce,group,force_insert=True)


    def get_length(self):
        return len(self.strategies)

    def get_stratey_list(self,device_choice,ps_or_reduce):
        new_device_array = device_choice
        ps_or_reduce = np.reshape(ps_or_reduce, (ps_or_reduce.shape[0], 1))
        new_device_array = np.concatenate((ps_or_reduce,new_device_array),axis=1)
        return new_device_array.tolist()

    def save_strategy_pool(self):
        if len(self.strategies)==0:
            return
        with open(self.folder_path + "/pool.pkl", "wb") as f:
            pkl.dump(list(self.strategies),f)
        with open(self.folder_path + "/pool.log", "w") as f:
            f.write(str([item["reward"] for item in self.strategies]))
            for i in range(4):
                f.write("\nstrategy"+str(i)+":\n")
                f.write(str(self.strategies[np.random.randint(len(self.strategies))]["strategy_list"]))


    def insert(self,reward,device_choice,replica_mask,ps_or_reduce,group,force_insert=False):
        def comp_fc(item):
            item1 = item[:int(len(item) / 2)]
            item2 = item[int(len(item) / 2):]
            return 0 if all(item1 == item2) else 1
        strategy_list = self.get_stratey_list(device_choice, ps_or_reduce)
        if force_insert:
            self.strategies.append({"replica_mask": replica_mask, "strategy_list": strategy_list, "reward": reward,
                                    "device_choice": device_choice, "ps_or_reduce": ps_or_reduce,"group":group})

            self.save_strategy_pool()
            self.rewards.append(reward)
            return

        if len(self.strategies)<20 and reward>np.mean(self.rewards):
            for j,strategy in enumerate(self.strategies):
                exist_device_choice = (strategy["device_choice"])
                if len(exist_device_choice)!=len(device_choice):
                    continue
                diff_list = list(map(comp_fc,np.concatenate((device_choice,exist_device_choice),axis=1)))
                if sum(diff_list)/len(diff_list)<0.2:
                    if reward>strategy["reward"]:
                        self.strategies.append({"replica_mask":replica_mask,"strategy_list":strategy_list,"reward":reward,"device_choice":device_choice,"ps_or_reduce":ps_or_reduce,"group":group})
                        self.strategies.pop(j)
                        self.save_strategy_pool()
                        self.rewards = [item["reward"] for item in self.strategies]
                    return
            self.strategies.append({"replica_mask": replica_mask, "strategy_list": strategy_list, "reward": reward,
                                    "device_choice": device_choice, "ps_or_reduce": ps_or_reduce,"group":group})

            self.save_strategy_pool()
            self.rewards.append(reward)
        elif len(self.strategies)>=10 and reward>np.mean(self.rewards):
            for j,strategy in enumerate(self.strategies):
                exist_device_choice = (strategy["device_choice"])
                if len(exist_device_choice)!=len(device_choice):
                    continue
                diff_list = list(map(comp_fc,np.concatenate((device_choice,exist_device_choice),axis=1)))
                if sum(diff_list)/len(diff_list)<0.2:
                    if reward>strategy["reward"]:
                        self.strategies.append({"replica_mask":replica_mask,"strategy_list":strategy_list,"reward":reward,"device_choice":device_choice,"ps_or_reduce":ps_or_reduce,"group":group})
                        self.strategies.pop(j)
                        self.save_strategy_pool()
                        self.rewards = [item["reward"] for item in self.strategies]
                    return
            index = self.rewards.index(min(self.rewards))
            self.strategies.pop(index)
            self.strategies.append({"replica_mask": replica_mask, "strategy_list": strategy_list, "reward": reward,
                                    "device_choice": device_choice, "ps_or_reduce": ps_or_reduce,"group":group})

            self.save_strategy_pool()
            self.rewards = [item["reward"] for item in self.strategies]

    def choose_strategy(self):
        if len(self.strategies)==0:
            return None
        self.rewards = [item["reward"] for item in self.strategies]
        index = np.random.randint(0,len(self.strategies))
        #index = self.rewards.index(max(self.rewards))
        return self.strategies[index]
def reward_func(item):
    new_device_array = np.zeros(shape=(len(devices)),dtype=np.int32)
    for j in range(len(item)):
        if item[j]!=-1 and item[j]!=len(devices):
            new_device_array[item[j]]+=1
    return new_device_array
class Environment(object):
    def __init__(self,gdef_path,devices,folder_path,batch_size,init_group,sink):

        self.gdef = graph_pb2.GraphDef()
        with open(gdef_path,"r")as f:
            txt = f.read()
        pbtf.Parse(txt,self.gdef)
        self.folder_path = folder_path
        self.random_strategy=list()
        self.best_strategy = mp.Manager().dict()
        self.best_strategy["time"] = sys.maxsize
        self.batch_size = batch_size
        self.devices =devices
        self.sink =sink
        self.init_group = init_group
        with open("nccl_model.pkl","rb") as f:
            self.nccl_model=pkl.load(f)

        bandwidth = config_dict.get("bandwidth",None)
        if bandwidth==None:
            self.intra = "5000"
            self.inter = "1250"
        else:
            self.intra = bandwidth[0]
            self.inter = bandwidth[1]

        self.null_gdef = graph_pb2.GraphDef()
        with open(folder_path+"/null_graph.pbtxt","r")as f:
            txt = f.read()
        pbtf.Parse(txt,self.null_gdef)
        self.name_cost_dict = self.get_name_cost_dict()
        if os.path.exists(folder_path+"/best_time.log"):
            with open(folder_path+"/best_time.log", "r") as f:
                tmp = json.load(f)
                for key,value in tmp.items():
                    self.best_strategy[key] = value
            with open(self.folder_path+"/best_time.log", "w") as f:

                cost_dict=dict()
                for key, value in self.name_cost_dict.items():
                    name = key[0]
                    replica_num=key[1]
                    if replica_num==1:
                        cost_dict[name] = value
                self.best_strategy["cost"] = cost_dict
                json.dump(self.best_strategy.copy(), f)
            _tge = tge.TGE(copy.deepcopy(self.null_gdef), self.devices, sink)
            time_mem_tuple = _tge.custom(self.best_strategy["strategy"]).fill_batchsize(self.batch_size).set_nccl_model(self.nccl_model).use_collective().set_bandwidth(self.intra, self.inter).evaluate(self.name_cost_dict,self.folder_path+"/best_graph.json")

            best_graph_def =tge.TGE(copy.deepcopy(self.null_gdef), self.devices, self.sink).custom(self.best_strategy["strategy"]).replace_placeholder(batch_size).use_collective().compile().get_result()
            with open(self.folder_path+"/best_graph.pbtxt", "w") as f:
                f.write(str(best_graph_def))





    def get_reward2(self,device_choice,ps_or_reduce,sink,group,record=False,record_name=None,record_best=True,from_strategy_pool=False):
        out_of_memory=False
        #new_device_array = np.zeros(shape=(device_choice.shape[0],len(devices)),dtype=np.int32)

        '''
        indexes = np.unique(group, return_index=True)[1]
        no_sort_group = [group[index] for index in sorted(indexes)]
        group = [no_sort_group.index(item) for item in group]
        '''
        new_device_array = device_choice
        ps_or_reduce = np.reshape(ps_or_reduce, (ps_or_reduce.shape[0], 1))
        new_device_array = np.concatenate((ps_or_reduce,new_device_array),axis=1)
        name_list = [nodedef.name for nodedef in self.null_gdef.node]
        print(new_device_array)
        strategy = {node.name:new_device_array[group[self.init_group[index]]].tolist() for index,node in enumerate(self.null_gdef.node)}
        strategy = {name: strategy.get(name, list(strategy.values())[0]) for name in name_list}

        _tge = tge.TGE(copy.deepcopy(self.null_gdef), self.devices,sink)

        time_mem_tuple = _tge.custom(strategy).fill_batchsize(self.batch_size).set_nccl_model(self.nccl_model).use_collective().set_bandwidth(self.intra,self.inter).evaluate(self.name_cost_dict)
        time = time_mem_tuple[0]
        mem_list = time_mem_tuple[1]
        time = float(time)/(10**3)

        if any(np.array(mem_list) > np.array(device_mems)):
            time = time*10
            out_of_memory=True
        #reward = np.sum(strategy*strategy)

        if time<self.best_strategy["time"] and out_of_memory==False and record_best:
            self.best_strategy["time"] = time
            self.best_strategy["strategy"] = strategy
            self.best_strategy["group"] = group
            with open(self.folder_path+"/best_time.log", "w") as f:

                cost_dict=dict()
                for key, value in self.name_cost_dict.items():
                    name = key[0]
                    replica_num=key[1]
                    if replica_num==1:
                        cost_dict[name] = value
                self.best_strategy["cost"] = cost_dict
                json.dump(self.best_strategy.copy(), f)

            best_graph_def = tge.TGE(copy.deepcopy(self.null_gdef), self.devices, self.sink).custom(strategy).replace_placeholder(self.batch_size).use_collective().compile().get_result()
            with open(self.folder_path+"/best_graph.pbtxt", "w") as f:
                f.write(str(best_graph_def))

        if record:
            record_graph_def = tge.TGE(copy.deepcopy(self.null_gdef), self.devices, self.sink).custom(strategy).replace_placeholder(self.batch_size).use_collective().compile().get_result()
            with open(self.folder_path+"/"+record_name, "w") as f:
                f.write(pbtf.MessageToString(record_graph_def))

        return -np.float32(np.sqrt(time)),out_of_memory

    def get_name_cost_dict(self):
        with open(self.folder_path+"/new_cost.pkl", "rb") as f:
            name_cost_dict = pkl.load(f)
        return name_cost_dict



class Graph_item():
    def __init__(self,folder_path,sink):
        with open(folder_path+"/feature.json","r") as f:
            feature_matrix = json.load(f)

        if "data/graph7" in folder_path:
            self.batch_size = 288*3
        elif "data/graph8" in folder_path:
            self.batch_size = 12 * 3
        else:
            self.batch_size = 36*3

        self.sink = sink
        if "graph1" in folder_path:
            self.master=True
        else:
            self.master = False

        ####preprocess features################
        feature_matrix = StandardScaler().fit_transform(feature_matrix)
        self.nb_nodes = feature_matrix.shape[0]
        '''
        self.pre_nb_nodes = feature_matrix.shape[0]
        pad_num = 0 if self.pre_nb_nodes%bsz==0 else bsz-self.pre_nb_nodes%bsz
        if pad_num:
            feature_matrix = np.pad(feature_matrix,((0,pad_num),(0,0)),"constant")
        self.nb_nodes = self.pre_nb_nodes+pad_num

        indptr = np.pad(adj.indptr, (0, pad_num), "edge")
        adj = csr_matrix((adj.data, adj.indices, indptr), shape=(self.nb_nodes,self.nb_nodes))
        '''
        self.ft_size = feature_matrix.shape[1]
        self.need_sample = False

        self.features =  tf.convert_to_tensor(feature_matrix, dtype=tf.float32)


        self.gdef = graph_pb2.GraphDef()
        with open(folder_path+"/null_graph.pbtxt","r")as f:
            txt = f.read()
        pbtf.Parse(txt,self.gdef)

        self.model_topo = dgl.DGLGraph()
        self.model_topo.add_nodes(len(self.gdef.node))
        reverse_dict = {node.name: i for i, node in enumerate(self.gdef.node)}
        for i, node in enumerate(self.gdef.node):
            for input in node.input:
                if input[0] == '^':
                    x = input[1:]
                else:
                    x = input.split(':')[0]
                self.model_topo.add_edge(i, reverse_dict[x])
                self.model_topo.add_edge(reverse_dict[x], i)
        self.model_topo.add_edges(self.model_topo.nodes(), self.model_topo.nodes())  # self-loops are required for GAT






        #####process init group#####
        if os.path.exists(folder_path+"/init_group.json"):
            with open(folder_path+"/init_group.json","r") as f:
                self.init_group =json.load(f)
        else:
           # self.init_group = tge.TGE(copy.deepcopy(self.gdef ), devices, sink).get_groups()
            self.init_group = self.get_colocation_group()
            with open(folder_path+"/new_cost.pkl", "rb") as f:
                name_cost_dict = pkl.load(f)
            self.init_group = group_around_topk_costs(self.gdef,self.init_group,name_cost_dict,group_num)
            with open(folder_path+"/init_group.json","w") as f:
                json.dump(self.init_group,f)
        #print(self.init_group)

        ########################create simulator###########################################
        self.env = Environment(folder_path+"/null_graph.pbtxt",devices,folder_path,self.batch_size,self.init_group,sink)
        self.average_reward=0
        self.best_reward = 1-sys.maxsize
        self.best_replica_num = list()
        self.best_device_choice = np.zeros(shape=(self.nb_nodes, len(devices)), dtype=np.int32)
        self.best_ps_or_reduce = list()
        self.folder_path = folder_path
        ##############create strategy pool#############################
        self.strategy_pool = strategy_pool(folder_path,self.nb_nodes,self.env,self.batch_size,self.init_group,self.sink)
        self.best_group= self.strategy_pool.choose_strategy()["group"] if self.strategy_pool.choose_strategy()!=None else np.arange(max(self.init_group)+1)
        self.avg = None
        self.oom = []
        self.train_place = False
        self.counter=0
        self.small_co = 0.001*5*5
        self.large_co =self.small_co*50
        self.co_entropy = self.small_co
        self.place_lr = lr
        self.record_time =[]
        self.mems = [np.zeros([128, bsz, d_model], dtype=np.float32) for layer in range(n_layer)]
    def get_colocation_group(self):
        leaders = []
        group = []
        for i, nodedef in enumerate(self.gdef.node):
            try:
                colocation_list = nodedef.attr["_class"].list.s
                if len(colocation_list)==0:
                    colocation_name = nodedef.name
                else:
                    colocation  = colocation_list[0]
                    colocation_name = colocation.decode().split("@")[-1]
                if colocation_name  not in leaders:
                    leaders.append(colocation_name)
                    group.append(len(leaders))
                else:
                    group.append(leaders.index(colocation_name))
            except Exception as e:
                traceback.print_exc()
                time.sleep(1)
        print(leaders)
        print(len(leaders))
        print(len(group))
        return group
    def set_network(self,place_gnn):
        self.place_gnn = place_gnn






    def sample(self,epoch):

        global sample_prob
        sample_prob = min(0.1+0.1*(epoch//60),0.7)

        print("[{}] sample_prob = {}".format(self.folder_path, sample_prob))

        self.replica_masks = mp.Manager().list(range(sample_times+1))
        self.device_choices = mp.Manager().list(range(sample_times+1))
        self.rewards = mp.Manager().list(range(sample_times+1))
        self.ps_or_reduces = mp.Manager().list(range(sample_times+1))
        self.group =mp.Manager().list(range(sample_times+1))
        self.oom = mp.Manager().list(range(sample_times+1))


        self.outputs = self.place_gnn.get_replica_num_prob(
            ftr_in=self.features,
            graph=self.model_topo,
            init_group = self.init_group)

    def parallel_process_output_unit(self,i):
        def random_func1(output):
            return np.array(list(map(random_choice, output)))

        def random_choice(item):
            np.random.seed()
            choice = []
            choice.append(np.random.choice(item.size, p=np.exp(item)))
            choice.append(np.random.randint(0, item.size))
            return choice[np.random.choice(2, p=[sample_prob, 1 - sample_prob])]

        def sample_func1(output):
            return np.array(list(map(sample_choice, output)))

        def sample_choice(item):
            return np.random.choice(item.size, p=np.exp(item))

        def argmax_func1(output):
            return np.array(list(map(argmax_choice, output)))

        def argmax_choice(item):
            choice1 = np.argmax(item)
            return choice1

        if i == sample_times:
            device_choice = np.array(list(map(argmax_func1, self.outputs[0:len(devices)])))
        else:
            np.random.seed()
            sample_or_not = True if np.random.choice(2, p=[sample_prob,1-sample_prob])==0 else False
            if sample_or_not:
                device_choice = np.array(list(map(sample_func1, self.outputs[0:len(devices)])))
            else:
                device_choice = np.array(list(map(random_func1, self.outputs[0:len(devices)])))
        print(device_choice.shape)
        device_choice = np.transpose(device_choice)  # from shape[device_num , group_num] to [group_num, device_num]
        device_choice, replica_mask = post_process_device_choice(device_choice, self.batch_size)

        if i == sample_times:
            ps_or_reduce = np.array(list(map(argmax_choice, self.outputs[len(devices)])))
        else:
            if sample_or_not:
                ps_or_reduce = np.array(list(map(sample_choice, self.outputs[len(devices)])))
            else:
                ps_or_reduce = np.array(list(map(random_choice, self.outputs[len(devices)])))
        # ps_or_reduce = self.outputs[max_replica_num]
        # group =  np.array(list(map(random_func1,self.outputs[-1])))
        
        for k in range(ps_or_reduce.shape[0]):
            replica_mask[k][ps_or_reduce[k]]=1
        
        
        _reward, out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.sink, self.init_group)
        if not out_of_memory:
            self.oom[i]=(False)
        else:
            self.oom[i]=(True)

        self.rewards[i]=(_reward)
        self.ps_or_reduces[i]=(ps_or_reduce)
        self.device_choices[i]=(device_choice)
        self.group[i]=(self.init_group)
        self.replica_masks[i]=(replica_mask)

    def parallel_process_output(self):
        self.thres = []
        for i in range(sample_times+1):
            p=mp.Process(target=self.parallel_process_output_unit, args=(i,))
            self.thres.append(p)
            p.start()
        for p in self.thres:
            p.join()

        #print("Group:",self.group[0])
    def post_parallel_process(self):
        for i in range(sample_times+1):
            if self.rewards[i] > self.best_reward:
                self.best_reward = self.rewards[i]
                self.best_replica_num = self.replica_masks[i]
                self.best_device_choice = self.device_choices[i]
                self.best_ps_or_reduce = self.ps_or_reduces[i]
                self.best_group = self.group[i]
            if not self.oom[i]:
                self.strategy_pool.insert(self.rewards[i], self.device_choices[i], self.replica_masks[i], self.ps_or_reduces[i],self.group[i])

    def compute_gradients(self,epoch):
        self.avg = np.mean(self.rewards) if self.avg==None else (self.avg+np.mean(self.rewards))/2
        print("[{}] train_place = {}".format(self.folder_path, self.train_place))
        print("[{}] Rewards = {}".format(self.folder_path, self.rewards))
        print("[{}] epoch = {}".format(self.folder_path, epoch))
        tmp_gradients = []

        results = [(mask,0.1*self.avg/reward) for mask,reward in zip(self.replica_masks,self.rewards)]
        results.pop()

        gradients=self.place_gnn.get_gradients(ftr_in=self.features,graph=self.model_topo,init_group=self.init_group,results=results)
        tmp_gradients.append(gradients)

        times = max(self.rewards)*max(self.rewards)

        if epoch % show_interval == 0:
            print("[{}] step = {}".format(self.folder_path,epoch))
            print("[{}] time = {}".format(self.folder_path,times))
            print("[{}] average reward = {}".format(self.folder_path,self.avg))
            with open(self.folder_path+"/time.log", "a+") as f:
                f.write(str(times) + ",")
            #with open(self.folder_path+"/entropy.log", "a+") as f:
             #   f.write(str(self.cal_entropy) + ",")
            #with open(self.folder_path+"/loss.log", "a+") as f:
             #   f.write("place loss:{},entropy loss:{},place+entropy loss:{},l2_loss:{}\n".format(place_loss,-self.cal_entropy*self.co_entropy,new_loss,l2_loss))

        if epoch % show_interval == 0:
            pool_strategy = self.strategy_pool.choose_strategy()
            if pool_strategy==None:
                return self.compute_average_gradients(tmp_gradients)
            results = [(pool_strategy["replica_mask"],0.1*self.avg/pool_strategy["reward"])]

            gradients = self.place_gnn.get_gradients(ftr_in=self.features, graph=self.model_topo,
                                                     init_group=self.init_group, results=results)

            print("time ratio:",0.1*self.avg/pool_strategy["reward"])
            tmp_gradients.append(gradients)

        return self.compute_average_gradients(tmp_gradients)
    def compute_average_gradients(self,tmp_gradients):
        for i,gradient in enumerate(tmp_gradients):
            if i == 0:
                # print type(actor_gradient), len(actor_gradient), type(actor_gradient[0]), len(actor_gradient[0])
                average_gradient = gradient
            else:
                for j in range(0, len(gradient)):
                    average_gradient[j] += gradient[j]
        for j in range(0, len(average_gradient)):
            average_gradient[j] = average_gradient[j] / len(tmp_gradients)
        return average_gradient


class new_place_GNN():
    def __init__(self,ft_size):
        self.first_time = True

        self.gat = GAT(ft_size,len(devices),max_replica_num)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9,beta_2=0.98, epsilon=1e-9)

        try:
            self.gat.load_weights('weights')
            print("load saved weight")
        except:
            print("no saved weight")
            pass
    def get_gradients(self,ftr_in,graph,init_group,results):
        self.gat.set_graphs(graph,init_group)
        with tf.GradientTape() as tape:
            tape.watch(self.gat.trainable_weights)
            logp = self.gat(ftr_in, training=True)
            reward = tf.add_n([reward_env * tf.reduce_sum(tf.boolean_mask(logp, mask)) for mask, reward_env in results])
            loss = -reward
            grads = tape.gradient(loss, self.gat.trainable_weights)
            grads = [tf.clip_by_value(grad, -1., 1.) for grad in grads]
        return grads
    def apply_gradients(self,grads):
        self.optimizer.apply_gradients(zip(grads, self.gat.trainable_weights))
    def get_replica_num_prob(self,ftr_in,graph,init_group):
        self.gat.set_graphs(graph, init_group)
        x = self.gat(ftr_in, training=True)
        x = x.numpy()
        outputs = [x[:,i*(max_replica_num+1):(i+1)*(max_replica_num+1)] for i in range(len(devices))]
        outputs.append(x[:,-2:])
        return outputs

def main_entry():
    models = []
    for i,feature_folder in enumerate(feature_folders):
        item = Graph_item(feature_folder,sinks[i])
        models.append(item)
    place_gnn = new_place_GNN(ft_size=models[0].ft_size)

    for model in models:
        model.set_network(place_gnn)


    for epoch in range(nb_epochs):
        for model in models:
            model.sample(epoch)

        processes=[]
        for model in models:
            processes.append(mp.Process(target=model.parallel_process_output))
            #model.parallel_process_output()
        for pro in processes:
            pro.start()
        for pro in processes:
            pro.join()

        gradients = []
        for model in models:
            model.post_parallel_process()
            #model.train(epoch)
            gradients.append(model.compute_gradients(epoch))

        for i, gradient in enumerate(gradients):
            if i == 0:
                average_gradient = gradient
            else:
                for j in range(0, len(gradient)):
                    average_gradient[j] += gradient[j]
        for j in range(0, len(average_gradient)):
            average_gradient[j] = average_gradient[j] / len(gradients)
        #print("Gradients:",average_gradient)
        place_gnn.apply_gradients(average_gradient)

        if epoch % (show_interval*30 )== 0:
            place_gnn.gat.save_weights('weights')


if __name__ == '__main__':

    main_entry()