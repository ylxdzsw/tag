from utils import load
import tge

d = {}

def f(m, o):
    gdef = load('raw_data/{}/model.pickle'.format(m))
    tge.simplify_graph(gdef, sinks=["Adam", "init"])
    for x in o:
        print(gdef.node[x].op)
        if gdef.node[x].op in d:
            d[gdef.node[x].op] += 1
        else:
            d[gdef.node[x].op] = 1

# {'Conv2DBackpropFilter': 66, 'BiasAdd': 19, 'ConcatV2': 2, 'Add': 26, 'Reshape': 341, 'MatMul': 336, 'AddN': 8, 'Transpose': 89, 'Pad': 4, 'VariableV2': 22, 'ExpandDims': 22, 'Conv2DBackpropInput': 11, 'ReluGrad': 11, 'UnsortedSegmentSum': 2, 'StridedSliceGrad': 2}
