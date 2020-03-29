import time
import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from models import GAT
from models import SpGAT
from gat_utils import process
from data_process.dataset import GraphDataset, WhiteSpaceTokenizer,NewWhiteSpaceTokenizer
from data_process.example import load_M10, load_cora, load_dblp
from data_process.meta_network import MetaNetwork, N_TYPE_NODE, N_TYPE_LABEL, IdIndexer
from transformer.model import transformer
import google.protobuf.text_format as pbtf
from tensorflow.core.framework import graph_pb2
from sklearn.preprocessing import StandardScaler
import copy
import sys
import os
import scipy.sparse as sp

import pickle
sys.path.append('../')
import tge
import json
import pickle as pkl
import multiprocessing as mp
from multiprocessing import Pool
from utils import adapt_batchsize

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
l2_coef = 0.2  # weight decay
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
group_num = 10
bsz =1

global_mems = [np.zeros([128, bsz, d_model], dtype=np.float32) for layer in range(n_layer)]


nonlinearity = tf.nn.elu
model = SpGAT
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
print('model: ' + str(model))
feature_folders = config_dict.get("inputs",["data/graph1", "data/graph2", "data/graph3", "data/graph4", "data/graph5", "data/graph6","data/graph7"])
sinks =  config_dict.get("sinks",[["GradientDescent"], ["GradientDescent"], ["GradientDescent"], ["GradientDescent"], ["GradientDescent"], ["GradientDescent"],["group_deps_1","loss/Mean","global_step/add"]])
sample_times = 3
devices = config_dict.get("devices", [
    "/job:worker/replica:0/task:0/device:GPU:0",
    "/job:worker/replica:0/task:0/device:GPU:1",
    "/job:worker/replica:0/task:1/device:GPU:0",
    "/job:worker/replica:0/task:1/device:GPU:1",
    "/job:worker/replica:0/task:2/device:GPU:0",
    "/job:worker/replica:0/task:2/device:GPU:1",
    "/job:worker/replica:0/task:3/device:GPU:0",
    "/job:worker/replica:0/task:3/device:GPU:1",
    "/job:worker/replica:0/task:4/device:GPU:0",
    "/job:worker/replica:0/task:4/device:GPU:1"

])

max_replica_num = config_dict.get("max_replica_num", len(devices))
show_interval = 3
num_cores = mp.cpu_count()
device_mems = config_dict.get("device_mems", [16 * 10e9, 16 * 10e9, 16 * 10e9, 16 * 10e9])
def find_index(array, item):
    for idx, val in enumerate(array):
        if val == item:
            return idx
    return len(array)
def post_func1(item):
    item1=item[:len(item)-1]
    batch_size = item[-1]
    replica_num = find_index(item1, len(devices))
    while(batch_size%replica_num):
        replica_num-=1
    if replica_num<len(item1):
        counter=0
        for i,elem in enumerate(item1):
            if counter>=replica_num:
                item1[i] = len(devices)
            if counter<replica_num and elem!=len(devices):
                counter+=1

    return item1
def post_func2(item1):
    replica_mask = np.array([0 if item==-1 else 1 for item in item1],dtype=np.int32)
    return replica_mask
def post_process_device_choice(device_choice,batch_size):

    #post process and align to batch size
    new_batch_size=np.ones(shape=(device_choice.shape[0],1)) * batch_size
    device_choice=np.array(list(map(post_func1,np.concatenate((device_choice,new_batch_size),axis=1))),dtype=np.int32)
    replica_mask = np.array(list(map(post_func2,device_choice)),dtype=np.int32)
    return device_choice,replica_mask

def comp_fc(item):
    item1 = item[:int(len(item)/2)]
    item2 = item[int(len(item)/2):]
    return 0 if all(item1 == item2) else 1
class strategy_pool(object):
    def __init__(self,folder_path,node_num,index_id_dict,env,batch_size,init_group,sink):
        self.folder_path = folder_path
        self.node_num = node_num
        self.index_id_dict = index_id_dict
        self.env = env
        self.sink = sink
        self.init_group = init_group
        self.init_group_num = max(self.init_group)+1
        if os.path.exists(self.folder_path+"/pool.pkl"):
            with open(self.folder_path+"/pool.pkl","rb") as f:
                self.strategies= pkl.load(f)
                for j, strategy in enumerate(self.strategies):
                    group = strategy["group"]
                    if len(group)!=self.init_group_num:
                        self.strategies.pop(j)
            self.save_strategy_pool()
        else:
            self.strategies = list()

        self.rewards = [item["reward"] for item in self.strategies] if len(self.strategies) else [-sys.maxsize]
        self.batch_size = batch_size

        # even data parallel 1
        #device_choice = np.zeros(shape=(self.node_num, len(devices)), dtype=np.int32)
        group = np.array([0 for i in range(self.init_group_num)],dtype=np.int32)
        device_choice = np.array([np.arange(max_replica_num)%(len(devices)) for i in range(group_num)],dtype=np.int32)
        device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
        ps_or_reduce = np.ones(shape=(group_num, ), dtype=np.int32)
        reward,out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink,group,record_best=False,from_strategy_pool=True)
        if not out_of_memory:
            self.insert(reward, device_choice, replica_mask,ps_or_reduce,group)

        # even data parallel 2
        #device_choice = np.zeros(shape=(self.node_num, len(devices)), dtype=np.int32)
        group = np.array([0 for i in range(self.init_group_num)],dtype=np.int32)
        device_choice = np.negative(np.ones(shape=(group_num, max_replica_num), dtype=np.int32))
        for item in device_choice:
            for i in range(len(devices)):
                item[i] =i
            item[len(devices):]=len(devices)

        device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
        ps_or_reduce = np.ones(shape=(group_num, ), dtype=np.int32)
        reward,out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink,group,record=True,record_name="nccl_dp_graph.pbtxt",record_best=False,from_strategy_pool=True)
        if not out_of_memory:
            self.insert(reward, device_choice, replica_mask,ps_or_reduce,group)


        # even data parallel 3
        #device_choice = np.zeros(shape=(self.node_num, len(devices)), dtype=np.int32)
        group = np.array([0 for i in range(self.init_group_num)],dtype=np.int32)
        device_choice = np.negative(np.ones(shape=(group_num, max_replica_num), dtype=np.int32))
        for item in device_choice:
            for i in range(len(devices)):
                item[i] =i
            item[len(devices):]=len(devices)

        device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
        ps_or_reduce = np.zeros(shape=(group_num, ), dtype=np.int32)
        reward,out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink,group,record=True,record_name="grpc_dp_graph.pbtxt",record_best=False,from_strategy_pool=True)
        if not out_of_memory:
            self.insert(reward, device_choice, replica_mask,ps_or_reduce,group)

        #single gpu
        group = np.array([0 for i in range(self.init_group_num)],dtype=np.int32)
        device_choice = np.negative(np.ones(shape=(group_num, max_replica_num), dtype=np.int32))
        for item in device_choice:
            item[0] =0
            item[1:] = len(devices)

        device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
        ps_or_reduce = np.ones(shape=(group_num, ), dtype=np.int32)
        reward,out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink,group,record=True,record_name="single_graph.pbtxt",record_best=False,from_strategy_pool=True)
        if not out_of_memory:
            self.insert(reward, device_choice, replica_mask,ps_or_reduce,group)

        #model parallel 1
        group = np.array([i%len(devices) for i in range(self.init_group_num)],dtype=np.int32)
        device_choice = np.negative(np.ones(shape=(group_num, max_replica_num), dtype=np.int32))
        for i,item in enumerate(device_choice):
            item[0] = i%(len(devices))
            item[1:] = len(devices)

        device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
        ps_or_reduce = np.ones(shape=(group_num, ), dtype=np.int32)
        reward,out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink,group,record_best=False,from_strategy_pool=True)
        if not out_of_memory:
            self.insert(reward, device_choice, replica_mask,ps_or_reduce,group)

        # model parallel 2
        group = np.array([int(i//(self.init_group_num/(len(devices)))) for i in range(self.node_num)],dtype=np.int32)
        device_choice = np.negative(np.ones(shape=(group_num, max_replica_num), dtype=np.int32))
        for i, item in enumerate(device_choice):
            item[0] = int(i//(len(device_choice)/(len(devices))))
            item[1:] = len(devices)

        device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
        ps_or_reduce = np.ones(shape=(group_num,), dtype=np.int32)
        reward, out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink,group,record_best=False,from_strategy_pool=True)
        if not out_of_memory:
            self.insert(reward, device_choice, replica_mask,ps_or_reduce,group)

        self.rewards = [item["reward"] for item in self.strategies]
    def get_length(self):
        return len(self.strategies)

    def get_stratey_list(self,device_choice,ps_or_reduce):
        new_device_array = np.array(list(map(reward_func,device_choice)),dtype=np.int32)
        ps_or_reduce = np.reshape(ps_or_reduce, (ps_or_reduce.shape[0], 1))
        new_device_array = np.concatenate((ps_or_reduce,new_device_array),axis=1)
        return new_device_array.tolist()

    def save_strategy_pool(self):
        with open(self.folder_path + "/pool.pkl", "wb") as f:
            pkl.dump(self.strategies,f)
        with open(self.folder_path + "/pool.log", "w") as f:
            f.write(str([item["reward"] for item in self.strategies]))
            for i in range(4):
                f.write("\nstrategy"+str(i)+":\n")
                f.write(str(self.strategies[np.random.randint(len(self.strategies))]["strategy_list"]))

    def get_replica_masks(self,device_choice):
        masks = np.zeros(shape=device_choice.shape,dtype=np.int32)
        for i in range(masks.shape[0]):
            for j in range(masks.shape[1]):
                masks[i,j] = 0 if device_choice[i,j]==-1 else 1
        return masks

    def insert(self,reward,device_choice,replica_mask,ps_or_reduce,group):
        strategy_list = self.get_stratey_list(device_choice, ps_or_reduce)

        if len(self.strategies)<4:
            for j,strategy in enumerate(self.strategies):
                exist_device_choice = (strategy["device_choice"])
                #diff_list = list(map(comp_fc,np.concatenate((device_choice,exist_device_choice),axis=1)))
                if False:#sum(diff_list)/len(diff_list)<0.05:
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

        elif len(self.strategies)<100 and reward>np.mean(self.rewards):
            for j,strategy in enumerate(self.strategies):
                exist_device_choice = (strategy["device_choice"])
                if len(exist_device_choice)!=len(device_choice):
                    continue
                diff_list = list(map(comp_fc,np.concatenate((device_choice,exist_device_choice),axis=1)))
                if sum(diff_list)/len(diff_list)<0.05:
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
        elif len(self.strategies)>=100 and reward>np.mean(self.rewards):
            for j,strategy in enumerate(self.strategies):
                exist_device_choice = (strategy["device_choice"])
                if len(exist_device_choice)!=len(device_choice):
                    continue
                diff_list = list(map(comp_fc,np.concatenate((device_choice,exist_device_choice),axis=1)))
                if sum(diff_list)/len(diff_list)<0.05:
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
        index = self.rewards.index(max(self.rewards))
        return self.strategies[index]
def reward_func(item):
    new_device_array = np.zeros(shape=(len(devices)),dtype=np.int32)
    for j in range(len(item)):
        if item[j]!=-1 and item[j]!=len(devices):
            new_device_array[item[j]]+=1
    return new_device_array
class Environment(object):
    def __init__(self,gdef_path,devices,folder_path,batch_size,pool,init_group,sink):

        self.gdef = graph_pb2.GraphDef()
        with open(gdef_path,"r")as f:
            txt = f.read()
        pbtf.Parse(txt,self.gdef)
        self.folder_path = folder_path
        self.random_strategy=list()
        self.best_strategy = mp.Manager().dict()
        self.best_strategy["time"] = sys.maxsize
        self.batch_size = batch_size
        self.pool = pool
        self.devices =devices
        self.sink =sink
        self.init_group = init_group
        with open("nccl_model.pkl","rb") as f:
            self.nccl_model=pkl.load(f)


        if "graph7" in folder_path:
            self.null_gdef =self.gdef
        else:
            self.null_gdef = graph_pb2.GraphDef()
            with open(folder_path+"/null_graph.pbtxt","r")as f:
                txt = f.read()
            pbtf.Parse(txt,self.null_gdef)

        if os.path.exists(folder_path+"/best_time.log"):
            with open(folder_path+"/best_time.log", "r") as f:
                tmp = json.load(f)
                for key,value in tmp.items():
                    self.best_strategy[key] = value
            best_graph_def =tge.TGE(copy.deepcopy(self.null_gdef), self.devices, self.sink).custom(self.best_strategy["strategy"]).replace_placeholder(batch_size).use_collective().compile().get_result()
            with open(self.folder_path+"/best_graph.pbtxt", "w") as f:
                f.write(str(best_graph_def))



        self.name_cost_dict = self.get_name_cost_dict()

    def get_reward2(self,device_choice,ps_or_reduce,index_id_dict,sink,group,record=False,record_name=None,record_best=True,from_strategy_pool=False):
        out_of_memory=False
        #new_device_array = np.zeros(shape=(device_choice.shape[0],len(devices)),dtype=np.int32)

        '''
        indexes = np.unique(group, return_index=True)[1]
        no_sort_group = [group[index] for index in sorted(indexes)]
        group = [no_sort_group.index(item) for item in group]
        '''
        group = group.tolist()
        new_device_array = np.array(list(map(reward_func,device_choice)))
        ps_or_reduce = np.reshape(ps_or_reduce, (ps_or_reduce.shape[0], 1))
        new_device_array = np.concatenate((ps_or_reduce,new_device_array),axis=1)
        strategy = {index_id_dict[index]:new_device_array[group[self.init_group[index]]].tolist() for index in range(len(index_id_dict))}
        bandwidth = config_dict.get("bandwidth",None)
        if bandwidth==None:
            intra = "5000"
            inter = "1250"
        else:
            intra = bandwidth[0]
            inter = bandwidth[1]
        _tge = tge.TGE(copy.deepcopy(self.gdef), self.devices,sink)

        time_mem_tuple = _tge.custom(strategy).replace_placeholder(self.batch_size).set_nccl_model(self.nccl_model).use_collective().set_bandwidth(intra,inter).evaluate(self.name_cost_dict)
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
                    batch_size=key[1]
                    if batch_size==self.batch_size:
                        cost_dict[name] = value
                self.best_strategy["cost"] = cost_dict
                json.dump(self.best_strategy.copy(), f)

            best_graph_def = tge.TGE(copy.deepcopy(self.null_gdef), self.devices, self.sink).custom(strategy).replace_placeholder(self.batch_size).use_collective().compile().get_result()
            with open(self.folder_path+"/best_graph.pbtxt", "w") as f:
                f.write(str(best_graph_def))

        if record:
            record_graph_def = tge.TGE(copy.deepcopy(self.null_gdef), self.devices, self.sink).custom(strategy).replace_placeholder(self.batch_size).use_collective().compile().get_result()
            with open(self.folder_path+"/"+record_name, "w") as f:
                f.write(str(record_graph_def))

        return -np.float32(np.sqrt(time)),out_of_memory

    def get_name_cost_dict(self):
        with open(self.folder_path+"/new_cost.pkl", "rb") as f:
            name_cost_dict = pkl.load(f)
        return name_cost_dict

sample_prob = 0.9
def random_choice(item):
    np.random.seed()
    choice1 = np.random.choice(item.size, p=item)
    #return choice1
    choice2 = np.random.randint(0,item.size)
    choices = [choice1,choice2]
    return choices[np.random.choice(2,p=[sample_prob,1-sample_prob])]

def sample_func1(output):
    return np.array(list(map(random_choice, output)))
def random_func1(item):
    return  random_choice(item)

def argmax_choice(item):
    choice1 = np.argmax(item)
    return choice1
def argmax_func1(output):
    return np.array(list(map(argmax_choice, output)))
def argmax_random_func1(item):
    return  argmax_choice(item)
import threading



class feature_item(threading.Thread):
    def __init__(self,folder_path,pool,event,event2,sink):
        super(feature_item, self).__init__()
        self.dataset = load_cora(folder_path,NewWhiteSpaceTokenizer())
        adj = self.dataset.adj_matrix(sparse=True)
        feature_matrix, feature_masks = self.dataset.feature_matrix(bag_of_words=False, sparse=False)
        self.batch_size = int(feature_matrix[0,-1])


        if self.batch_size<global_batch_size:
            self.batch_size=global_batch_size
        else:
            self.batch_size = int((self.batch_size//global_batch_size)*global_batch_size)


        self.event = event
        self.event.set()
        self.event2 = event2
        self.event2.clear()
        self.sink = sink
        if "graph1" in folder_path:
            self.master=True
        else:
            self.master = False
        feature_matrix = StandardScaler().fit_transform(feature_matrix)

        labels, label_masks = self.dataset.label_list_or_matrix(one_hot=False)

        train_node_indices, test_node_indices, train_masks, test_masks = self.dataset.split_train_and_test(training_rate=0.8)

        self.pre_nb_nodes = feature_matrix.shape[0]
        pad_num = 0 if self.pre_nb_nodes%bsz==0 else bsz-self.pre_nb_nodes%bsz
        if pad_num:
            feature_matrix = np.pad(feature_matrix,((0,pad_num),(0,0)),"constant")
        self.nb_nodes = self.pre_nb_nodes+pad_num

        indptr = np.pad(adj.indptr, (0, pad_num), "edge")
        adj = csr_matrix((adj.data, adj.indices, indptr), shape=(self.nb_nodes,self.nb_nodes))

        self.ft_size = feature_matrix.shape[1]
        self.need_sample = False

        self.biases = process.preprocess_adj_bias(adj)

        train_mask=train_masks
        val_mask=test_masks
        test_mask=test_masks

        self.index_id_dict = self.dataset.network.get_indexer(N_TYPE_NODE).index_id_dict
        self.features = feature_matrix[np.newaxis]

        self.train_mask = train_mask[np.newaxis]
        self.val_mask = val_mask[np.newaxis]
        self.test_mask = test_mask[np.newaxis]

        self.pool = pool
        self.gdef = graph_pb2.GraphDef()
        with open(folder_path+"/graph.pbtxt","r")as f:
            txt = f.read()
        pbtf.Parse(txt,self.gdef)
        self.init_group = tge.TGE(copy.deepcopy(self.gdef ), devices, sink).get_groups()
        print(self.init_group)
        self.env = Environment(folder_path+"/graph.pbtxt",devices,folder_path,self.batch_size,self.pool,self.init_group,sink)
        self.average_reward=0
        self.best_reward = 1-sys.maxsize
        self.best_replica_num = list()
        self.best_device_choice = np.zeros(shape=(self.nb_nodes, max_replica_num), dtype=np.int32)
        self.best_ps_or_reduce = list()
        self.folder_path = folder_path

        self.strategy_pool = strategy_pool(folder_path,self.nb_nodes,self.index_id_dict,self.env,self.batch_size,self.init_group,self.sink)
        self.best_group= self.strategy_pool.choose_strategy()["group"] if self.strategy_pool.choose_strategy()!=None else np.arange(max(self.init_group)+1)

        self.mutex = threading.Lock()
        self.avg = 0
        self.oom = []
        self.train_place = False
        self.counter=0


    def set_session_and_network(self,sess,place_gnn):
        self.sess =sess
        self.place_gnn = place_gnn

    def sample_one_time(self):
        self.sample()
        self.proc = mp.Process(target=self.parallel_process_output, args=())
        self.proc.start()
    def wait_sample(self):
        self.proc.join()

    def sync_sample_and_parallel_process(self):
        self.sample()
        self.parallel_process_output()

    def sample(self):

        self.replica_masks = mp.Manager().list(range(sample_times+1))
        self.device_choices = mp.Manager().list(range(sample_times+1))
        self.rewards = mp.Manager().list(range(sample_times+1))
        self.ps_or_reduces = mp.Manager().list(range(sample_times+1))
        self.group =mp.Manager().list(range(sample_times+1))
        self.oom = mp.Manager().list(range(sample_times+1))
        co_entropy = 0
        self.thres = []

        self.outputs = self.place_gnn.get_replica_num_prob_and_entropy(
            ftr_in=self.features,
            bias_in=self.biases,
            nb_nodes=self.nb_nodes,
            mems=global_mems,
            sample_group=np.array(self.best_group),
            init_group = self.init_group)

    def parallel_process_output_unit(self,i):
        if i == sample_times:
            device_choice = np.array(list(map(argmax_func1, self.outputs[0:max_replica_num])))
        else:
            device_choice = np.array(list(map(sample_func1, self.outputs[0:max_replica_num])))
        # device_choice = self.outputs[0:max_replica_num]
        device_choice = np.transpose(device_choice)

        device_choice, replica_mask = post_process_device_choice(device_choice, self.batch_size)
        if i == sample_times:
            ps_or_reduce = np.array(list(map(argmax_random_func1, self.outputs[max_replica_num])))
        else:
            ps_or_reduce = np.array(list(map(random_func1, self.outputs[max_replica_num])))
        # ps_or_reduce = self.outputs[max_replica_num]
        # group =  np.array(list(map(random_func1,self.outputs[-1])))
        group = self.outputs[-1]
        _reward, out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict, self.sink, group)
        if not out_of_memory:
            self.oom[i]=(False)
        else:
            self.oom[i]=(True)

        self.rewards[i]=(_reward)
        self.ps_or_reduces[i]=(ps_or_reduce)
        self.device_choices[i]=(device_choice)
        self.group[i]=(group)
        self.replica_masks[i]=(replica_mask)

    def parallel_process_output(self):
        for i in range(sample_times+1):
            '''
            if i==sample_times:
                device_choice = np.array(list(map(argmax_func1, self.outputs[0:max_replica_num])))
            else:
                device_choice = np.array(list(map(sample_func1, self.outputs[0:max_replica_num])))
            #device_choice = self.outputs[0:max_replica_num]
            device_choice = np.transpose(device_choice)

            device_choice, replica_mask = post_process_device_choice(device_choice, self.batch_size)
            if i==sample_times:
                ps_or_reduce = np.array(list(map(argmax_random_func1, self.outputs[max_replica_num])))
            else:
                ps_or_reduce = np.array(list(map(random_func1, self.outputs[max_replica_num])))
            #ps_or_reduce = self.outputs[max_replica_num]
            #group =  np.array(list(map(random_func1,self.outputs[-1])))
            group =self.outputs[-1]
            _reward, out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink,group)
            if not out_of_memory:
                self.oom.append(False)
            else:
                self.oom.append(True)

            self.rewards.append(_reward)
            self.ps_or_reduces.append(ps_or_reduce)
            self.device_choices.append(device_choice)
            self.group.append(group)
            self.replica_masks.append(replica_mask)
            '''
            p=mp.Process(target=self.parallel_process_output_unit, args=(i,))
            self.thres.append(p)
            p.start()
        for p in self.thres:
            p.join()

        print("Group:",self.group[0])
    def post_parallel_process(self):
        for i in range(sample_times+1):
            self.cal_entropy = self.outputs[-2]
            if self.rewards[i] > self.best_reward:
                self.best_reward = self.rewards[i]
                self.best_replica_num = list()
                self.best_device_choice = self.device_choices[i]
                self.best_ps_or_reduce = self.ps_or_reduces[i]
                self.best_group = self.group[i]
            if not self.oom[i]:
                self.strategy_pool.insert(self.rewards[i], self.device_choices[i], self.replica_masks[i], self.ps_or_reduces[i],self.group[i])

    def train(self,epoch):
        global global_mems
        tr_step = 0
        co_entropy = 10
        tr_size = self.features.shape[0]
        '''
        if self.strategy_pool.get_length()>0:
            pool_strategy = self.strategy_pool.choose_strategy()
            self.rewards.append(pool_strategy["reward"])
            self.device_choices.append(pool_strategy["device_choice"])
            self.ps_or_reduces.append((pool_strategy["ps_or_reduce"]))
            self.replica_masks.append(pool_strategy["replica_mask"])
            self.group.append(pool_strategy["group"])
        '''
        self.avg = float(np.mean(self.strategy_pool.rewards)) if self.strategy_pool.get_length()>0 else np.mean(self.rewards)
        if self.master:
            global sample_prob
            if sample_prob<0.9:
                sample_prob+=0.01
            if len(set(self.rewards))==1:
                sample_prob=0.7

        print("[{}] train_place = {}".format(self.folder_path, self.train_place))
        print("[{}] Rewards = {}".format(self.folder_path, self.rewards))


        for index in range(sample_times):
            new_loss,new_global_mems=self.place_gnn.learn(ftr_in=self.features,
                            bias_in=self.biases,
                            nb_nodes=self.nb_nodes,
                            replica_num_array=np.array(self.replica_masks[index]),
                            sample_ps_or_reduce = np.array(self.ps_or_reduces[index]),
                            sample_device_choice = np.array(self.device_choices[index]),
                            sample_group=np.array(self.group[index]),
                            time_ratio = ((self.rewards[index])-self.avg)/np.abs(self.avg),
                            coef_entropy=co_entropy,
                            mems = global_mems,
                            init_group=self.init_group)
            global_mems = new_global_mems
        '''
        for i in range(sample_times):
            if self.oom[i] == False:
                self.avg = (self.avg+self.rewards[i])/2
        '''

        times = max(self.rewards)*max(self.rewards)

        if epoch % show_interval == 0:
            print("[{}] step = {}".format(self.folder_path,epoch))
            print("[{}] time = {}".format(self.folder_path,times))
            print("[{}] average reward = {}".format(self.folder_path,self.avg))
            print("[{}] overall entropy:{}".format(self.folder_path,self.cal_entropy))
            with open(self.folder_path+"/time.log", "a+") as f:
                f.write(str(times) + ",")
            with open(self.folder_path+"/entropy.log", "a+") as f:
                f.write(str(self.cal_entropy) + ",")
            with open(self.folder_path+"/loss.log", "a+") as f:
                f.write(str(new_loss) + ",")

        if epoch % show_interval == 0:
            return
            pool_strategy = self.strategy_pool.choose_strategy()
            if pool_strategy==None:
                return
            new_loss,new_global_mems=self.place_gnn.learn(ftr_in=self.features,
                            bias_in=self.biases,
                            nb_nodes=self.nb_nodes,
                            replica_num_array=np.array(pool_strategy["replica_mask"]),
                            sample_ps_or_reduce = np.array(pool_strategy["ps_or_reduce"]),
                            sample_device_choice = np.array(pool_strategy["device_choice"]),
                            sample_group=np.array(pool_strategy["group"]),
                            time_ratio = ((pool_strategy["reward"])-self.avg)/np.abs(self.avg),
                            coef_entropy=co_entropy,
                            mems = global_mems,
                            init_group=self.init_group)
            global_mems = new_global_mems
                


    def run(self):
        while True:
            self.event.wait()
            self.sample()
            self.process_output()
            self.event.clear()
            self.event2.set()


class new_place_GNN():
    def __init__(self,sess,ft_size):

        with tf.name_scope('place_gnn'):
            self.sess = sess

            self.ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, ft_size),name="ftr_in")
            #self.bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, None),name="bias_in")
            self.bias_in = tf.sparse_placeholder(dtype=tf.float32)
            self.nb_node = tf.placeholder(dtype=tf.int32, shape=(),name="nb_node")
            self.attn_drop = tf.placeholder(dtype=tf.float32, shape=(),name="attn_drop")
            self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=(),name="ffd_drop")
            self.is_train = tf.placeholder(dtype=tf.bool, shape=(),name="is_train")
            self.init_group = tf.placeholder(dtype=tf.int32, shape=(None,),name="init_group")
            self.sample_ps_or_reduce = tf.placeholder(dtype=tf.int32, shape=(None,),name="sample_ps_or_reduce")
            self.sample_device_choice = tf.placeholder(dtype=tf.int32, shape=(None,max_replica_num,),name="sample_device_choice")
            self.sample_group = tf.placeholder(dtype=tf.int32, shape=(None,),name="sample_group")
            self.replica_num_array = tf.placeholder(dtype=tf.float32, shape=(None,max_replica_num),name="replica_num_array")
            self.time_ratio = tf.placeholder(dtype=tf.float32, shape=(),name="time_ratio")
            self.coef_entropy = tf.placeholder(dtype=tf.float32, shape=(),name="coef_entropy")
            self.train_place = tf.placeholder(dtype=tf.bool, shape=(),name="train_place")
            self.mems = [tf.placeholder(tf.float32,[128, bsz, d_model]) for _ in range(n_layer)]

        with tf.variable_scope("group_nn"):
            with tf.device("/device:GPU:0"):
                group = model.inference(self.ftr_in, group_num, self.nb_node, self.is_train,
                                         0,0,
                                         bias_mat=self.bias_in,
                                         hid_units=hid_units, n_heads=n_heads,
                                         residual=residual, activation=nonlinearity)
            group =tf.reshape(group, [-1,group_num])
            group =tf.unsorted_segment_max(group, self.init_group,tf.reduce_max(self.init_group)+1)
            #_,sample_group = tf.unique(self.sample_group)


            self.pro_group = tf.nn.softmax(group)
            self.log_pro_group = tf.log(self.pro_group+10e-8)
            log_pro_group = tf.nn.log_softmax(group)
            self.group = tf.random.categorical(log_pro_group,1,dtype=tf.int32)
            self.group = tf.reshape(self.group,[-1])
            #_,real_group = tf.unique(self.group)
            group = tf.cond(self.is_train,lambda:self.sample_group,lambda:self.group)
            unique_group,_ =tf.unique(group)
            #group = tf.cond(self.is_train,lambda:tf.unique(self.sample_group[0])[0],lambda:group)

            self.group_entropy = tf.reduce_sum(self.pro_group * log_pro_group, 1)
            self.group_entropy = -tf.reduce_mean(self.group_entropy)

            _range1 = tf.range(tf.shape(self.sample_group)[0])[:, tf.newaxis]

            # one_hot_sample = tf.one_hot(self.sample_ps_or_reduce[i], 2)
            # print("one_hot_sample.shape")
            # print(one_hot_sample.shape)
            # prob = tf.reduce_sum( self.ps_or_reduce * one_hot_sample, 1)
            self.indices = tf.concat((_range1, self.sample_group[:, tf.newaxis]), axis=1)
            self.log_prob = tf.gather_nd(self.log_pro_group, self.indices)
            self.group_loss = tf.reduce_sum(self.log_prob) * self.time_ratio

        with tf.variable_scope("place_nn"):
            with tf.device("/device:GPU:1"):
                logits = model.inference(self.ftr_in, 1024, self.nb_node, self.is_train,
                                         self.attn_drop, self.ffd_drop,
                                         bias_mat=self.bias_in,
                                         hid_units=hid_units, n_heads=n_heads,
                                         residual=residual, activation=nonlinearity)

            with tf.device("/device:GPU:0"):
                logits = model.inference(logits, d_model, self.nb_node, self.is_train,
                                         self.attn_drop, self.ffd_drop,
                                         bias_mat=self.bias_in,
                                         hid_units=place_hid_units, n_heads=place_n_heads,
                                         residual=residual, activation=nonlinearity)
            logits = tf.reshape(logits, [-1, d_model])
            logits = tf.unsorted_segment_max(logits, self.init_group, tf.reduce_max(self.init_group) + 1)
            logits = tf.reshape(logits, [-1, d_model])
            self.logits_before = logits
            logits=tf.unsorted_segment_sum(logits, group,group_num)
            self.logits =logits
            self.device_choices = list()
            self.log_device_choices = list()
            log_resh = tf.reshape(logits, [-1,bsz,d_model])
            initializer = tf.initializers.random_normal(
                stddev=0.02,
                seed=None)
            output,self.new_mems = transformer(log_resh,self.mems,n_layer,d_model,n_head,d_head,d_inner,0.1,0.1,initializer,True,mem_len=128)
            output = tf.reshape(output, [-1,d_model])
            output = tf.layers.dense(output,max_replica_num*(len(devices)+1)+1)

            sum = 0
            o1 = tf.nn.softmax(output[:,0:len(devices)])
            self.device_choices.append(o1)
            #log_o1 = tf.nn.log_softmax(output[:,0:len(devices)])
            log_o1 = tf.log(o1+10e-8)
            self.log_device_choices.append(log_o1)

            sum = sum + tf.reduce_mean(tf.reduce_sum(log_o1 * o1, 1))
            for i in range(1,max_replica_num):
                oi = tf.nn.softmax(output[:,i*(len(devices)+1)-1:(i+1)*(len(devices)+1)-1])
                self.device_choices.append(oi)
                #log_oi = tf.nn.log_softmax(output[:,i*(len(devices)+1)-1:(i+1)*(len(devices)+1)-1])
                log_oi = tf.log(oi+10e-8)
                self.log_device_choices.append(log_oi)

                sum = sum + tf.reduce_mean(tf.reduce_sum(log_oi* oi, 1))
            ps_or_reduce_prob = tf.nn.softmax(output[:,-2:])
            self.ps_or_reduce = ps_or_reduce_prob
            #self.log_ps_reduce = tf.nn.log_softmax(output[:,-2:])
            self.log_ps_reduce = tf.log( self.ps_or_reduce+10e-8)
            self.entropy = tf.reduce_sum(self.log_ps_reduce * ps_or_reduce_prob, 1)
            self.entropy = -(tf.reduce_mean(self.entropy) + sum / max_replica_num)

            _range = tf.range(tf.shape(self.sample_ps_or_reduce)[0])[:, tf.newaxis]



            self.place_loss = []
            #one_hot_sample = tf.one_hot(self.sample_group[i], group_num)
            #print("one_hot_sample.shape")
            #print(one_hot_sample.shape)
            #prob = tf.reduce_sum( self.pro_group * one_hot_sample, 1)
            indices = tf.concat((_range, self.sample_ps_or_reduce[:, tf.newaxis]), axis=1)
            log_prob = tf.gather_nd(self.log_ps_reduce, indices)
            log_prob = tf.gather(log_prob,unique_group)
            self.place_loss.append(tf.reduce_sum(log_prob )* self.time_ratio)

            #first device choice n*m
            #one_hot_sample = tf.one_hot(self.sample_device_choice[i][:, 0], len(devices))
            #prob = tf.reduce_sum(self.device_choices_prob[0] * one_hot_sample, 1) * self.replica_num_array[i][:, 0] + (1 - self.replica_num_array[i][:, 0])


            self.indices = tf.concat((_range, self.sample_device_choice[:, 0][:, tf.newaxis]), axis=1)
            self.log_prob = tf.gather_nd(self.log_device_choices[0], self.indices)
            self.log_prob = tf.gather(self.log_prob,unique_group)
            self.place_loss.append(tf.reduce_sum(self.log_prob) * self.time_ratio)

            #rest device choice n*(m+1)
            for j in range(1,max_replica_num):
                #one_hot_sample = tf.one_hot(self.sample_device_choice[i][:,j], len(devices)+1)
                #prob = tf.reduce_sum(self.device_choices_prob[j] * one_hot_sample, 1) * self.replica_num_array[i][:,j]+(1-self.replica_num_array[i][:,j])
                indices = tf.concat((_range, self.sample_device_choice[:, j][:, tf.newaxis]), axis=1)
                log_prob = tf.gather_nd(self.log_device_choices[j], indices)
                log_prob = tf.gather(log_prob, unique_group)
                self.place_loss.append(tf.reduce_sum(log_prob) * self.time_ratio)
            self.place_loss = tf.add_n(self.place_loss)

        #self.place_loss = 100*self.place_loss
        place_reward =  self.place_loss+self.coef_entropy * self.entropy
        group_reward = (self.group_loss+self.coef_entropy*self.group_entropy)
        reward = place_reward+group_reward
        self.loss = -reward
        place_loss = -place_reward
        group_loss = -group_reward
        train_place = model.training(place_loss, lr, l2_coef,vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='place_nn'))
        train_group =model.training(group_loss, lr, 0,vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='group_nn'))
        #self.train_op =tf.cond(self.train_place,lambda:train_place,lambda:train_group)
        self.train_op =tf.group(train_place,train_group)


        '''
        for op in tf.get_default_graph().get_operations():
            if op.node_def.op in variable_ops:
                op._set_device("/device:CPU:0")

        '''

    def get_a_cell(self):
        return tf.nn.rnn_cell.BasicLSTMCell(num_units=64)

    def learn(self,ftr_in,bias_in,nb_nodes,replica_num_array,sample_ps_or_reduce,sample_device_choice,sample_group,time_ratio,coef_entropy,mems,init_group):
        feed_dict = {}
        feed_dict[self.ftr_in]=ftr_in
        feed_dict[self.bias_in]=bias_in
        feed_dict[self.nb_node]=nb_nodes
        feed_dict[self.is_train]=True
        feed_dict[self.attn_drop]=0
        feed_dict[self.ffd_drop]=0
        feed_dict[self.replica_num_array]=replica_num_array
        feed_dict[self.sample_ps_or_reduce]=sample_ps_or_reduce
        feed_dict[self.sample_device_choice]=sample_device_choice
        feed_dict[self.sample_group] = sample_group
        feed_dict[self.time_ratio]=time_ratio
        feed_dict[self.coef_entropy]=coef_entropy
        feed_dict[self.init_group]=init_group

        for item1,item2 in zip(self.mems,mems):
            feed_dict[item1]=item2
        log_device_choices,log_prob,pro_group,indices,loss,mems,_ = self.sess.run([self.log_device_choices[0],self.log_prob,self.pro_group,self.indices,self.loss,self.new_mems,self.train_op],
                     feed_dict=feed_dict)
        print("device_choice_log1:",log_device_choices)
        print("Indices:", indices)
        print("gather Log prob:", log_prob)
        print("Time ratio:",time_ratio)



        return loss,mems
    def get_replica_num_prob_and_entropy(self,ftr_in,bias_in,nb_nodes,mems,sample_group,init_group):
        fetch_list =[item for item in self.device_choices]
        fetch_list.append(self.ps_or_reduce)
        fetch_list.append(self.logits_before)
        fetch_list.append(self.logits)
        fetch_list.append(self.group)
        fetch_list.append(self.entropy)
        fetch_list.append(self.group)



        feed_dict = {}
        feed_dict[self.ftr_in]=ftr_in
        feed_dict[self.bias_in]=bias_in
        feed_dict[self.nb_node]=nb_nodes
        feed_dict[self.is_train]=False
        feed_dict[self.sample_group]=sample_group
        feed_dict[self.attn_drop]=0
        feed_dict[self.ffd_drop]=0
        feed_dict[self.init_group] = init_group
        for item1,item2 in zip(self.mems,mems):
            feed_dict[item1]=item2

        outputs = self.sess.run(fetch_list, feed_dict=feed_dict)
        print("device choice prob:", outputs[0])
        #print("Logits after:",outputs[-4])
        #print("Group:",outputs[-3])
        return outputs


def architecture_three():
    global  global_pool,models
    for i,feature_folder in enumerate(feature_folders):
        event = threading.Event()
        event2 = threading.Event()

        item = feature_item(feature_folder,global_pool,event,event2,sinks[i])
        item.setDaemon(True)
        models.append(item)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess:
        place_gnn = new_place_GNN(sess,ft_size=models[0].ft_size)

        saver = tf.train.Saver()
        try:
            saver.restore(sess, checkpt_file)
        except:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

        for model in models:
            model.set_session_and_network(sess,place_gnn)


        for epoch in range(nb_epochs):
            for model in models:
                #model.sample_one_time()
                #model.wait_sample()
                model.sync_sample_and_parallel_process()
                model.post_parallel_process()
                model.train(epoch)

            '''
            for model in models:
                model.sample_one_time()

            for model in models:
                model.wait_sample()
            for model in models:
                model.post_parallel_process()

            for model in models:
                model.train(epoch)
            '''
            if epoch % show_interval == 0:
                saver.save(sess, checkpt_file)

        sess.close()

import signal
def handler(signum, frame):
    global  global_pool,models
    try:
        for model in models:
            model.proc.terminate()
    except:
        pass
    sys.exit()

if __name__ == '__main__':

    models = list()
    global_pool = Pool(1)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    architecture_three()
                                                                                                                                                                                      