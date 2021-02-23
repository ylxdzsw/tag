from utils import save, load, info
from tge import simplify_graph

x = load("records")[0]['gdef']
len(x.node)

simplify_graph(x, ["Adam"])
len(x.node)
