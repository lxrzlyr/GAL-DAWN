import networkx as nx
import numpy as np
import argparse
import scipy


# 读取mtx格式的矩阵文件，返回为稀疏矩阵的coo表示格式


def read_mtx(mtx_file_path):
    sparse_matrix = scipy.io.mmread(mtx_file_path)
    G = nx.from_scipy_sparse_array(sparse_matrix, create_using=nx.DiGraph)
    return G


# 创建解析器对象并定义命令行参数
parser = argparse.ArgumentParser(
    description="Calculate shortest path length from source node in mtx graph file")
parser.add_argument("input_file", metavar="INPUT_FILE",
                    type=str, help="path to input mtx graph file")
parser.add_argument("output_file", metavar="OUTPUT_FILE",
                    type=str, help="path to output file")

# 解析命令行参数
args = parser.parse_args()

G = read_mtx(args.input_file)

# 介数中心性
bc_value = nx.betweenness_centrality(G, normalized=True, weight=None)
# 对所有节点按节点编号进行排序
sorted_nodes = sorted(bc_value.keys())
with open(args.output_file, "w") as f:
    for node in sorted_nodes:
        value = bc_value[node]
        if value > 0:
            f.write("{} {:.6f}\n".format(node, value))