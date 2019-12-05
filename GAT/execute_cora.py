import time
import numpy as np
import tensorflow as tf

from models import GAT
from models import SpGAT
from utils import process
from data_process.dataset import GraphDataset, WhiteSpaceTokenizer,NewWhiteSpaceTokenizer
from data_process.example import load_M10, load_cora, load_dblp
from data_process.meta_network import MetaNetwork, N_TYPE_NODE, N_TYPE_LABEL, IdIndexer
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

checkpt_file = 'pre_trained/cora/mod_cora.ckpt'
_dataset = 'cora'

# training params
batch_size = 1
nb_epochs = 100000
patience = 100
lr = 0.01  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [64,32] # numbers of hidden units per each attention head in each layer
n_heads = [8,8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = SpGAT

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
feature_folders = ["data/graph1","data/graph2"]
sample_times = 10
devices = (
    "/job:tge/replica:0/task:0/device:GPU:0",
    "/job:tge/replica:0/task:0/device:GPU:1",
    "/job:tge/replica:0/task:1/device:GPU:0",
    "/job:tge/replica:0/task:1/device:GPU:1"
)
show_interval = 10





class Environment(object):
    def __init__(self,gdef_path,devices,folder_path):

        self.gdef = graph_pb2.GraphDef()
        with open(gdef_path,"r")as f:
            txt = f.read()
        pbtf.Parse(txt,self.gdef)
        self.folder_path = folder_path
        self.random_strategy=list()
        self.best_time = sys.maxsize
        self.best_strategy = dict()
        self.strategy_time_dict=dict()

        if os.path.exists(folder_path+"/best_time.log"):
            with open(folder_path+"/best_time.log", "r") as f:
                tmp = json.load(f)
                self.best_time = tmp["time"]
                self.best_strategy = tmp["strategy"]
                self.strategy_time_dict[str(self.best_strategy)] = self.best_time


        self.devices =devices
        self.name_cost_dict = self.get_name_cost_dict()
        self._tge = tge.TGE(self.gdef, devices)

    def get_reward2(self,strategy,device_choice,ps_or_reduce,index_id_dict):

        new_device_array = np.zeros(shape=device_choice.shape,dtype=np.int32)
        for i in range(device_choice.shape[0]):
            for j in range(device_choice.shape[1]):
                if device_choice[i,j]!=-1 and device_choice[i,j]!=len(self.devices):
                    new_device_array[i,device_choice[i,j]]+=1
        ps_or_reduce = np.reshape(ps_or_reduce, (ps_or_reduce.shape[0], 1))
        new_device_array = np.concatenate((ps_or_reduce,new_device_array),axis=1)
        strategy = {index_id_dict[index]:new_device_array[index].tolist() for index in range(new_device_array.shape[0])}
        time = tge.TGE(copy.deepcopy(self.gdef), self.devices).custom(strategy).evaluate(self.name_cost_dict)
        time = float(time)/(10**6)
        #reward = np.sum(strategy*strategy)
        if time<self.best_time:
            self.best_time = time
            self.best_strategy = strategy
            with open(self.folder_path+"/best_time.log", "w") as f:
                tmp = dict()
                tmp["time"] = time
                tmp["strategy"] = self.best_strategy
                tmp["cost"] = self.name_cost_dict
                json.dump(tmp, f)

        return -np.float32(np.sqrt(time))

    def get_name_cost_dict(self):
        name_cost_dict = dict()
        with open(self.folder_path+"/docs.txt", "r") as f:
            for line in f.readlines():
                line = line.strip()
                items = line.split(" ")
                name = items[0]
                cost = list(np.array(items[-len(devices):]))
                name_cost_dict[name] = cost
        return name_cost_dict



class feature_item(object):
    def __init__(self,folder_path):
        self.dataset = load_cora(folder_path,NewWhiteSpaceTokenizer())
        adj = self.dataset.adj_matrix(sparse=True)
        feature_matrix, feature_masks = self.dataset.feature_matrix(bag_of_words=False, sparse=False)

        feature_matrix = StandardScaler().fit_transform(feature_matrix)

        labels, label_masks = self.dataset.label_list_or_matrix(one_hot=False)

        train_node_indices, test_node_indices, train_masks, test_masks = self.dataset.split_train_and_test(training_rate=0.8)

        self.nb_nodes = feature_matrix.shape[0]
        self.ft_size = feature_matrix.shape[1]

        adj = adj.todense()
        adj = adj[np.newaxis]
        self.biases = process.adj_to_bias(adj, [self.nb_nodes], nhood=5)
        self.biases = self.biases[0]
        self.biases = process.preprocess_adj_bias(sp.csr_matrix(self.biases))

        train_mask=train_masks
        val_mask=test_masks
        test_mask=test_masks

        self.index_id_dict = self.dataset.network.get_indexer(N_TYPE_NODE).index_id_dict
        self.features = feature_matrix[np.newaxis]

        self.train_mask = train_mask[np.newaxis]
        self.val_mask = val_mask[np.newaxis]
        self.test_mask = test_mask[np.newaxis]


        self.env = Environment(folder_path+"/graph.pbtxt",devices,folder_path)
        self.average_reward=0
        self.best_reward = 1-sys.maxsize
        self.best_replica_num = list()
        self.best_replica_n_hot_num = np.zeros(shape=(self.nb_nodes, len(devices)), dtype=np.int32)
        self.best_device_choice = np.zeros(shape=(self.nb_nodes, len(devices)), dtype=np.int32)
        self.best_ps_or_reduce = list()
        self.folder_path = folder_path



    def set_session_and_network(self,sess,place_gnn):
        self.sess =sess
        self.place_gnn = place_gnn

    def sample_and_train(self,epoch):
        co_entropy = 1000

        tr_step = 0
        tr_size = self.features.shape[0]
        replica_n_hot_nums = list()
        device_choices = list()
        rewards = list()
        ps_or_reduces=list()
        for i in range(sample_times):
            replica_num = list()
            replica_n_hot_num = np.zeros(shape=(self.nb_nodes,len(devices)),dtype=np.int32)
            device_choice = np.zeros(shape=(self.nb_nodes,len(devices)),dtype=np.int32)
            ps_or_reduce=list()
            outputs = self.place_gnn.get_replica_num_prob_and_entropy(ftr_in=self.features[tr_step * batch_size:(tr_step + 1) * batch_size],
                                                                      bias_in=self.biases,
                                                                 nb_nodes=self.nb_nodes)

            finished_node = list()
            for i in range(len(devices)):
                output = outputs[i]
                for j,pred in enumerate(output):
                    if j not in finished_node:
                        index = np.random.choice(pred.size,p=pred)
                        if index==len(devices):
                            finished_node.append(j)
                            device_choice[j, i] = index
                            replica_n_hot_num[j,i]=1
                        else:
                            device_choice[j,i] = index
                            replica_n_hot_num[j,i]=1
                    else:
                        device_choice[j,i] = -1
            device_choices.append(device_choice)
            replica_n_hot_nums.append(replica_n_hot_num)

            for j, pred in enumerate(outputs[len(devices)]):
                index = np.random.choice(pred.size, p=pred)
                ps_or_reduce.append(index)
            ps_or_reduce = np.array(ps_or_reduce)
            ps_or_reduces.append(ps_or_reduce)

            cal_entropy = outputs[-1]
            _reward = self.env.get_reward2(replica_num, device_choice,ps_or_reduce,self.index_id_dict)
            if _reward>self.best_reward:
                self.best_reward = _reward
                self.best_replica_num = replica_num
                self.best_replica_n_hot_num = replica_n_hot_num
                self.best_device_choice = device_choice
                self.best_ps_or_reduce = ps_or_reduce


            rewards.append(_reward)
        if epoch>20:
            self.average_reward = (self.average_reward*19+np.mean(rewards))/20
        else:
            self.average_reward = (self.average_reward * epoch+ np.mean(rewards)) /(epoch+1)


        while tr_step * batch_size < tr_size:
            new_loss=self.place_gnn.learn(ftr_in=self.features[tr_step * batch_size:(tr_step + 1) * batch_size],
                            bias_in=self.biases,
                            nb_nodes=self.nb_nodes,
                            replica_num_array=np.array(replica_n_hot_nums),
                            sample_ps_or_reduce = np.array(ps_or_reduces),
                            sample_device_choice = np.array(device_choices),
                            time_ratio = (np.array(rewards)-float(self.average_reward)),
                            coef_entropy=co_entropy)
            tr_step += 1

        if epoch % show_interval == 0:
            print("[{}] step = {}".format(self.folder_path,epoch))
            print("[{}] time = {}".format(self.folder_path,rewards*rewards))
            print("[{}] average reward = {}".format(self.folder_path,self.average_reward))
            print("[{}] overall entropy:{}".format(self.folder_path,cal_entropy))
            with open(self.folder_path+"/time.log", "a+") as f:
                f.write(str(np.mean(rewards*rewards)) + ",")
            with open(self.folder_path+"/entropy.log", "a+") as f:
                f.write(str(cal_entropy) + ",")
            with open(self.folder_path+"/loss.log", "a+") as f:
                f.write(str(new_loss) + ",")




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
            self.sample_ps_or_reduce = tf.placeholder(dtype=tf.int32, shape=(None,None,),name="sample_ps_or_reduce")
            self.sample_device_choice = tf.placeholder(dtype=tf.int32, shape=(None,None,len(devices),),name="sample_device_choice")
            self.replica_num_array = tf.placeholder(dtype=tf.float32, shape=(None,None,len(devices)),name="replica_num_array")
            self.time_ratio = tf.placeholder(dtype=tf.float32, shape=(None,),name="time_ratio")
            self.coef_entropy = tf.placeholder(dtype=tf.float32, shape=(),name="coef_entropy")

        logits = model.inference(self.ftr_in, 64, self.nb_node, self.is_train,
                                 self.attn_drop, self.ffd_drop,
                                 bias_mat=self.bias_in,
                                 hid_units=hid_units, n_heads=n_heads,
                                 residual=residual, activation=nonlinearity)
        log_resh = tf.reshape(logits, [-1, 64])
        log_resh = tf.layers.dense(log_resh, units=64, activation=tf.nn.relu)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([self.get_a_cell() for _ in range(len(devices))])
        h0 = self.cell.zero_state(self.nb_node, np.float32)
        self.output,self.h = self.cell.call(log_resh, h0)
        self.ps_or_reduce = tf.layers.dense(self.output, units=2, activation=tf.nn.softmax)
        self.device_choices = list()
        sum=0
        out = tf.layers.dense(self.h[0].c, units=len(devices), activation=tf.nn.softmax)
        self.device_choices.append(out)
        sum = sum + tf.reduce_mean(tf.reduce_sum(tf.log(out + np.power(10.0, -9)) * out, 1))
        for i in range(len(devices)-1):
            out = tf.layers.dense(self.h[i+1].c, units=len(devices)+1, activation=tf.nn.softmax)
            self.device_choices.append(out)
            sum = sum+tf.reduce_mean(tf.reduce_sum(tf.log(out + np.power(10.0, -9)) *out, 1))

        self.entropy = tf.reduce_sum(tf.log(self.ps_or_reduce + np.power(10.0, -9)) * self.ps_or_reduce, 1)
        self.entropy = tf.reduce_mean(self.entropy)+sum/len(devices)
        for i in range(sample_times):
            one_hot_sample = tf.one_hot(self.sample_ps_or_reduce[i], 2)
            print("one_hot_sample.shape")
            print(one_hot_sample.shape)

            prob = tf.reduce_sum( self.ps_or_reduce * one_hot_sample, 1)
            # prob = tf.reduce_prod(peirob)
            reward = tf.reduce_sum(tf.log(prob + np.power(10.0, -9)) * self.time_ratio[i])

            #first device choice n*m
            one_hot_sample = tf.one_hot(self.sample_device_choice[i][:, 0], len(devices))
            prob = tf.reduce_sum(self.device_choices[0] * one_hot_sample, 1) * self.replica_num_array[i][:, 0] + (
                        1 - self.replica_num_array[i][:, 0])
            reward += tf.reduce_sum(tf.log(prob + np.power(10.0, -9)) * self.time_ratio[i])

            #rest device choice n*(m+1)
            for j in range(1,len(devices)):
                one_hot_sample = tf.one_hot(self.sample_device_choice[i][:,j], len(devices)+1)
                prob = tf.reduce_sum(self.device_choices[j] * one_hot_sample, 1) * self.replica_num_array[i][:,j]+(1-self.replica_num_array[i][:,j])
                reward+=tf.reduce_sum(tf.log(prob + np.power(10.0, -9)) * self.time_ratio[i])


        self.loss = -reward/sample_times #+ self.coef_entropy * self.entropy
        self.train_op = model.training(self.loss, lr, l2_coef)



    def get_a_cell(self):
        return tf.nn.rnn_cell.BasicLSTMCell(num_units=64)

    def learn(self,ftr_in,bias_in,nb_nodes,replica_num_array,sample_ps_or_reduce,sample_device_choice,time_ratio,coef_entropy):
        loss,_ = self.sess.run([self.loss,self.train_op],
                     feed_dict={
                         self.ftr_in: ftr_in,
                         self.bias_in:bias_in,
                         self.nb_node:nb_nodes,
                         self.is_train: True,
                         self.attn_drop: 0.1, self.ffd_drop: 0.1, self.replica_num_array: replica_num_array,
                         self.sample_ps_or_reduce:sample_ps_or_reduce,
                         self.sample_device_choice:sample_device_choice,
                         self.time_ratio:time_ratio, self.coef_entropy: coef_entropy})
        return loss
    def get_replica_num_prob_and_entropy(self,ftr_in,bias_in,nb_nodes):
        fetch_list =[item for item in self.device_choices]
        fetch_list.append(self.ps_or_reduce)
        fetch_list.append(self.entropy)
        outputs = self.sess.run(fetch_list, feed_dict={
            self.ftr_in: ftr_in,
            self.bias_in: bias_in,
            self.nb_node:nb_nodes,
            self.is_train: True,
            self.attn_drop: 0.1, self.ffd_drop: 0.1})
        return outputs

def architecture_three():
    models = list()
    for feature_folder in feature_folders:
        models.append(feature_item(folder_path=feature_folder))
    with tf.Session() as sess:
        place_gnn = new_place_GNN(sess,ft_size=models[0].ft_size)

        for model in models:
            model.set_session_and_network(sess,place_gnn)

        saver = tf.train.Saver()
        try:
            saver.restore(sess, checkpt_file)
        except:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

        for epoch in range(nb_epochs):

            for model in models:
                model.sample_and_train(epoch)

            if epoch % show_interval == 0:
                saver.save(sess, checkpt_file)

        sess.close()

architecture_three()