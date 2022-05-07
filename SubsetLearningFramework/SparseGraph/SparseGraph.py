import networkx as nx
import pickle
from utility.util import timestamp, mkdir


# we build a edge between x and y, iff {x, y} appear in some subset of data
def graph1(data, node_num) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(node_num))
    for subset in data:
        for x in subset:
            for y in subset:
                if x > y:
                    g.add_edge(x, y)
    return g


# -----------------------------------------------------------------------------------------------------
# need change
task_index = "4"
build_graph = graph1
# -----------------------------------------------------------------------------------------------------

f = open("../../tasks/task"+task_index+"/info.pickle", "rb")
info = pickle.load(f)
f.close()

time_str = timestamp()
time_str = time_str.replace(':', '_')
mkdir("../../tasks/task"+task_index+"/SparseGraph")
mkdir("../../tasks/task"+task_index+"/SparseGraph/"+time_str)

pre_path = "../../tasks/task"+task_index+"/SparseGraph/"+time_str

log_file = open(pre_path+"/log.txt", 'w')
log_file.write("The build method we used is " + build_graph.__name__ + "\n")
log_file.write("There are " + str(info['#node']) + " nodes in the graph" + "\n")
log_file.write("There are " + str(info['#group']) + " groups" + "\n")
log_file.write("The edges number of each group as follows" + "\n")


for group_index in range(int(info['#group'])):
    data = pickle.load(open("../../tasks/task"+task_index+"/train"+str(group_index)+".pickle", 'rb'))
    g = build_graph(data, int(info['#node']))
    pickle.dump(g, open(pre_path+"/graph"+str(group_index)+".pickle", 'wb'))
    print(len(g.edges))
    log_file.write(str(len(g.edges)) + "\n")
