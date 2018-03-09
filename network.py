import pandas
import pandas as pd
from pandas import DataFrame,Series
import os
import numpy as np
import networkx as nx
from igraph import Graph
import types
import igraph
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import collections
import seaborn as sns

file_name = "/Users/lizy/Downloads/Q3/Complex_Network/assignment/Data_Highschool.txt"

def read_file():
    row = 0
    Data_Highschool = pandas.read_table(file_name, delim_whitespace=True, names=('i', 'j', 't'))
    # print(Data_Highschool.shape)#188509,3
    return Data_Highschool


def properties():
    number_of_nodes = Data_Highschool.groupby(['i', 'j']).ngroups  # 5818
    Frequency_links = Data_Highschool.groupby(['i', 'j']).size().reset_index(name="Frequency")
    return number_of_nodes, Frequency_links


def find_duplicate_pairs():
    No_frequency_links = Frequency_links[:-1]
    only_nodes = No_frequency_links.iloc[:, 0:2].copy()
    nodes_list_pairs = only_nodes.values.tolist()
    No_duplicate_links = {tuple(item) for item in map(sorted, nodes_list_pairs)}
    number_of_links = len(No_duplicate_links)
    return number_of_links, only_nodes, No_duplicate_links

def create_graph():
    g = nx.Graph()
    duplicated_nodes_list = only_nodes.iloc[:, 0].append(only_nodes.iloc[:, 1]).reset_index(drop=True)
    nodes_list = duplicated_nodes_list.values.tolist()
    No_duplicate_nodes = set(nodes_list)
    print(len(No_duplicate_nodes))  # 327
    g.add_nodes_from(No_duplicate_nodes)
    g.add_edges_from(No_duplicate_links)
    nx.draw(g,node_size = 1.5)#with_labels=True
    plt.draw()
    plt.show()
    link_density = nx.density(g)
    #print(link_density)#0.109
    average_degree = nx.average_neighbor_degree(g)
    # print(average_degree)
    degree_correlation = nx.degree_pearson_correlation_coefficient(g)
    # print(degree_correlation)#0.033175769936049336
    average_clustering = nx.average_clustering(g)
    # print(average_clustering)#0.5035048191728447
    average_hopcount = nx.average_shortest_path_length(g)
    # print(average_hopcount)#2.1594341569576554
    diameter = nx.diameter(g)
    # print(diameter)#4
    # A = nx.adjacency_matrix(g)
    A_eigenvalue = nx.adjacency_spectrum(g)
    # print(max(A_eigenvalue))#(41.231605032525835+0j)
    G_eigenvalue = nx.laplacian_spectrum(g)
    # print(sorted(G_eigenvalue))#1.9300488624481513
    return g, nodes_list, No_duplicate_nodes, link_density, average_degree

file_name = file_name
Data_Highschool = read_file()

number_of_nodes, Frequency_links = properties()

number_of_links, only_nodes, No_duplicate_links = find_duplicate_pairs()

nodes_list, g, No_duplicate_nodes, link_density, average_degree = create_graph()

G = nx.Graph()
G.add_nodes_from(No_duplicate_nodes)
G.add_edges_from(No_duplicate_links)



Data_0 = Data_Highschool[Data_Highschool.t=='1']
#list_I_0 = Data_0.i.drop_duplicates() #出现在t=1时的i点
list_I_0 = list(G.nodes)
#list_seed = map(eval, list_I_0)
#list_I_0.index = range(len(list_I_0))

list_num_infected_nodes = [] #记录了以N个点为seed node时，在每个T steps的infected nodes的个数

for i in range(len(list_I_0)):

    seed = [list_I_0[i]]
    set_I = set(seed)
    print(set_I)
    length = [len(set_I)]

    for t in range(1,int(list(Data_Highschool.t)[-1])+1):
        a = []
        data = Data_Highschool[Data_Highschool.t == str(t)]

        for index in data.index:

            if set([data.i[index]]).issubset(set_I):
                #print("new infected:" + data.i[index])
                a.append(data.j[index])

        a = set(a)
        set_I = set_I.union(a)
        length.append(len(set_I))

    list_num_infected_nodes.append(length)
    print("len of infected nodes when seed is" + list_I_0[i] + ":" + str(len(set_I)))

fileObject = open('/Users/lizy/Desktop/Number_of_infected_nodes_all.txt', 'w')
for length in list_num_infected_nodes:
    for len_t in length:
        fileObject.write(str(len_t))
        fileObject.write(',')
    fileObject.write('\n')
fileObject.close()

matrix = np.array(list_num_infected_nodes).T #转化为T*N的矩阵
matrix = np.delete(matrix,1,0)
mean_T = np.mean(matrix, axis=1) # 计算每一行的均值
variance_T = np.std(matrix, axis=1)

x = range(1,int(list(Data_Highschool.t)[-1])+1)
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.plot(x, mean_T,label="Mean",color="red",linewidth=2)
axes.plot(x, variance_T,label="Variance",color="blue",linewidth=1)
axes.set_xlabel("Step(t)")
axes.set_title("Average number of infected nodes and its variance")
axes.legend(loc=4)
plt.show()


#calculate influence(the shortest time to reach 80% of the total nodes)
find = 0.8*G.number_of_nodes()
shortest_step = []
for i in range(len(list_I_0)):
    length = list_num_infected_nodes[i]
    step = [ind for ind,v in enumerate(length) if v>=find]
    if step!=[]:
        shortest_step.append(step[0])
    else:
        shortest_step.append(10000)

df_shortest_step = DataFrame({"node":list(G.nodes),"shortest_time":shortest_step})
df_influence = df_shortest_step.sort_values(by='shortest_time')
df_influence.index = range(len(df_influence))

# explore the correlation between network features(degree and clustering coefficient) and influence

#degree
nodes_deg_list = [n for n, d in G.degree()]
degree_list = [d for n, d in G.degree()]

degree_sequence = DataFrame({"node":nodes_deg_list,"degree":degree_list})

#clustering coefficient
nodes_clu_list = nx.clustering(G).keys()
cluster_list = nx.clustering(G).values()
cluster_sequence = DataFrame({"node":nodes_clu_list,"cluster_coefficient":cluster_list})

R_D = pd.merge(df_influence,degree_sequence,on=['node'],how='outer')
R_D_C = pd.merge(R_D,cluster_sequence,on=['node'],how='outer')

R_D_C.to_csv("/Users/lizy/Downloads/Q3/Complex_Network/assignment/R_degree_cluster.csv")
