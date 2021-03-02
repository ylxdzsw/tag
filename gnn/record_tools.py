from utils import load
import tge
import google.protobuf.text_format as pbtf

records = load("records")

def write_pbtxt():
    for m in ("resnet", "vgg", "transformer", "bert", "inception"):
        gdef, _, _, _ = load("{}_1080ti.pickle".format(m))
        tge.simplify_graph(gdef, sinks=["Adam"])
        with open("{}.pbtxt".format(m), "w") as fo:
            fo.write(pbtf.MessageToString(gdef))

if __name__ == "__main__":
    import sys
    eval(sys.argv[1])()
