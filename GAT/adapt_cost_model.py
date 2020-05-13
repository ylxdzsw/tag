import pickle as pkl
import sys
import multiprocessing as mp
sys.path.append('../')
from utils import adapt_batchsize
from profiler import Profiler
import tensorflow as tf
def adapt_cost(folder,init_batch,after_batch,max_replica):
    with open(folder+"cost.pkl","rb") as f:
        name_cost_dict = pkl.load(f)
    name_cost_dict = adapt_batchsize(name_cost_dict,init_batch,after_batch,max_replica)
    with open(folder+"new_cost.pkl","wb") as f:
        pkl.dump(name_cost_dict,f)



models = ["vgg19","resnet200","resnet50","resnet101","resnet152","inceptionv3","transformer","bert"]
processes = []
for i in range(len(models)):
    if i==6:
        tf.reset_default_graph()
        folder = "data/graph"+str(i+1)+"/"
        #adapt_cost(folder,288,288*3,18)
        processes.append(mp.Process(target=adapt_cost,args=(folder,288,288*2,18,)))
    if i==7:
        tf.reset_default_graph()
        folder = "data/graph"+str(i+1)+"/"
        #adapt_cost(folder,12,12*3,18)
        processes.append(mp.Process(target=adapt_cost,args=(folder,12,18,18,)))

    else:
        tf.reset_default_graph()
        folder = "data/graph"+str(i+1)+"/"
        #adapt_cost(folder,36,36*3,18)
        processes.append(mp.Process(target=adapt_cost,args=(folder,36,36*2,18,)))

for process in processes:
    process.start()
for process in processes:
    process.join()
