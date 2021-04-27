from utils import load
from data import ProfileData, estimate_model_size
import numpy as np
import tge

for m in ("inception", "resnet", "vgg", "transformer", "bert", "berts", "rnnlm2x", "rnnlm4x"): # ("inception", "resnet", "vgg", "transformer", "bert", "rnnlm2x", "rnnlm4x"): #  , "mobilenet", "nasnet"
    gdef = load('raw_data/{}/model.pickle'.format(m))
    prof_data = ProfileData(m)

    model_size = estimate_model_size(gdef, prof_data.maximum_batchsize())

    print("{}: number_of_nodes:{} parameter_size:{} model_size (with Adam states):{}".format(m, len(gdef.node), model_size >> 22, model_size >> 20))
