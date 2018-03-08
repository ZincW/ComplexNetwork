import pandas
import pandas as pd
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

fileObject = open('/Users/lizy/Desktop/Number_of_infected_nodes.txt', 'w')
for length in list_num_infected_nodes:
    for len_t in length:
        fileObject.write(str(len_t))
        fileObject.write(',')
    fileObject.write('\n')
fileObject.close()


mean = []
var = []
for t in range(1,int(list(Data_Highschool.t)[-1])+1):
    sum = 0
    variance = 0
    for i in range(len(list_I_0)):
        sum = sum + list_num_infected_nodes[i][t]
    sum = sum/float(len(list_I_0))
    mean.append(sum)

    for i in range(len(list_I_0)):
        variance = variance + (list_num_infected_nodes[i][t] - sum) ** 2
    variance = (variance/float(len(list_I_0)))**0.5
    var.append(variance)

