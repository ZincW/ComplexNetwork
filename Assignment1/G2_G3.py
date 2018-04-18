import pandas
import pandas as pd
from pandas import DataFrame,Series
import os
import numpy as np
import networkx as nx
from igraph import Graph
import types
import pandas
import igraph
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import collections
import seaborn as sns
import random

file_name = "/Users/lizy/Downloads/Q3/Complex_Network/assignment/Data_Highschool.txt"

def read_file():
    row = 0
    Data_Highschool = pandas.read_table(file_name, delim_whitespace=True, names=('i', 'j', 't'))
    # print(Data_Highschool.shape)#188509,3
    return Data_Highschool

file_name = file_name
Data_Highschool = read_file()

Data_Highschool = Data_Highschool.drop([0])

Data_Highschool.index = range(len(Data_Highschool))

Data_Highschool.t = map(eval, Data_Highschool.t)


#G2 -- randomlized G


T = Data_Highschool.t.copy()


random.shuffle(T)

#T.insert(0,'T')

new_Data_Highschool = Data_Highschool.copy()
new_Data_Highschool['T'] = T

new_Data_Highschool = new_Data_Highschool.sort_values(by='T')

new_Data_Highschool.to_csv('/Users/lizy/Downloads/Q3/Complex_Network/assignment/G2.csv')


nodes = new_Data_Highschool.iloc[:,0:2].copy()

nodes_list_pairs = nodes.values.tolist()

links = [tuple(item) for item in map(sorted, nodes_list_pairs)]


G = nx.Graph()

# G.add_nodes_from()
G.add_edges_from(links)


#G3

num_chosen_links = random.randint(0,len(Data_Highschool))

#slice = random.sample(Data_Highschool.index, num_chosen_links)

G3_Data_Highschool = Data_Highschool.copy()
G3_Data_Highschool = G3_Data_Highschool.sample(n=num_chosen_links).sort_values(by='t')
G3_Data_Highschool.index = range(len(G3_Data_Highschool))
G3_Data_Highschool.to_csv('/Users/lizy/Downloads/Q3/Complex_Network/assignment/G3.csv')

