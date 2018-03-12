import pandas
import pandas as pd
from pandas import DataFrame
import os
import numpy as np
import networkx as nx
from igraph import Graph
import types
import igraph
import matplotlib
import scipy.integrate as spi
import matplotlib.pyplot as plt 
import matplotlib.pylab as pylab
import collections
import seaborn as sns
import pylab as pl
from math import floor
import statistics
import json
from collections import defaultdict

class network:

	def __init__(self, file_name = "G2.csv"):
		self.file_name = file_name
		self.Data_Highschool = self.read_file()
		self.number_of_nodes, self.Frequency_links = self.properties()
		self.number_of_links, self.only_nodes, self.No_duplicate_links = self.find_duplicate_pairs()
		self.nodes_list, self.g, self.No_duplicate_nodes, self.link_density, self.average_degree = self.create_graph()
		
		# self.density_graph()
		self.mean_infected_dict, self.std_infected_dict = self.sis_model()
		self.plot()
		# self.random_graph()

	def read_file(self):
		row = 0 
		Data_Highschool = pandas.read_csv(self.file_name)
		# , delim_whitespace=True
		# names=('i', 'j', 't')
		print(Data_Highschool.shape)#188509,3
		return Data_Highschool

	def properties(self):
		number_of_nodes = self.Data_Highschool.groupby(['i', 'j']).ngroups #5818
		Frequency_links = self.Data_Highschool.groupby(['i', 'j']).size().reset_index(name="Frequency")
		return number_of_nodes, Frequency_links


	def find_duplicate_pairs(self):
		No_frequency_links = self.Frequency_links[:-1]
		only_nodes = No_frequency_links.iloc[:,0:2].copy()
		nodes_list_pairs = only_nodes.values.tolist()
		No_duplicate_links = {tuple(item) for item in map(sorted, nodes_list_pairs)}
		number_of_links = len(No_duplicate_links)
		# print(number_of_links)
		return number_of_links, only_nodes, No_duplicate_links

		################################### Part 1 ##########################################
	def create_graph(self):
		g = nx.Graph()
		duplicated_nodes_list = self.only_nodes.iloc[:,0].append(self.only_nodes.iloc[:,1]).reset_index(drop=True)
		nodes_list = duplicated_nodes_list.values.tolist()
		No_duplicate_nodes = set(nodes_list)
		# print(len(No_duplicate_nodes))#327
		g.add_nodes_from(No_duplicate_nodes)
		g.add_edges_from(self.No_duplicate_links)
		# nx.draw(g,node_size = 1.5)#with_labels=True
		# plt.draw()
		# plt.show()
		link_density = nx.density(g)
		# print(link_density)#0.109
		average_degree = nx.average_neighbor_degree(g)
		# numbers degreeede= [average_degree[key] for key in average_degree]
		# mean = statistics.mean(numbers)
		# var = statistics.variance(numbers)
		# print(var)
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



	# def density_graph(self):
	# 	g = nx.Graph()
	# 	duplicated_nodes_list = self.only_nodes.iloc[:,0].append(self.only_nodes.iloc[:,1]).reset_index(drop=True)
	# 	nodes_list = duplicated_nodes_list.values.tolist()
	# 	No_duplicate_nodes = set(nodes_list)
	# 	g.add_nodes_from(No_duplicate_nodes)
	# 	g.add_edges_from(self.No_duplicate_links)
	# 	degree_sequence = sorted([d for n, d in g.degree()], reverse=True)  # degree sequence
	# 	degreeCount = collections.Counter(degree_sequence)
	# 	deg, cnt = zip(*degreeCount.items())
	# 	fig, ax = plt.subplots()
	# 	plt.bar(deg, cnt, width=0.8, color='b')
	# 	plt.title("Degree Histogram")
	# 	plt.ylabel("Count")
	# 	plt.xlabel("Degree")
	# 	ax.set_xticks([d + 0.6 for d in deg])
	# 	ax.set_xticklabels(deg,fontsize=8)
	# 	# axes.labelsize: medium 
	# 	# matplotlib.rcParams.update({'font.size': 7})


	# 	# draw graph in inset
	# 	plt.axes([0.4, 0.4, 0.5, 0.5])
	# 	Gcc = sorted(nx.connected_component_subgraphs(g), key=len, reverse=True)[0]
	# 	pos = nx.spring_layout(g)
	# 	plt.axis('off')
	# 	nx.draw_networkx_nodes(g, pos, node_size=20)
	# 	nx.draw_networkx_edges(g, pos, alpha=0.4)

	# 	plt.show()
	# 	return 

	# def random_graph(self):
	# 	G = nx.gnp_random_graph(327, 0.1)
	# 	degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
	# 	degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
	# 	degreeCount = collections.Counter(degree_sequence)
	# 	deg,cnt = zip(*degreeCount.items())
	# 	fig, ax = plt.subplots()
	# 	plt.bar(deg, cnt, width=0.80, color='b')

	# 	plt.title("Degree Histogram")
	# 	plt.ylabel("Count")
	# 	plt.xlabel("Degree")
	# 	ax.set_xticks([d + 0.4 for d in deg])
	# 	ax.set_xticklabels(deg)

	# 	# draw graph in inset
	# 	# plt.axff[0.4, 0.4, 0.5, 0.5])
	# 	# Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
	# 	# pos = nx.spring_layout(G)
	# 	# plt.axis('off')
	# 	# nx.draw_networkx_nodes(G, pos, node_size=20)
	# 	# nx.draw_networkx_edges(G, pos, alpha=0.4)

	# 	plt.show()
	# 	return 

		############################# Question 9 #############################
	

	def sis_model(self):
		history_dict = rec_dd()
		for node in self.No_duplicate_nodes:
			infected_nodes = set([int(node)])
			for index, row in self.Data_Highschool.iterrows():
				if row['i'] in infected_nodes:
					infected_nodes.add(row['j'])
				elif row['j'] in infected_nodes:
					infected_nodes.add(row['i'])
				history_dict[row['t']][node] = len(infected_nodes)
			print('We have all infected nodes of node:',node)
			df_G2 = DataFrame()
			np.save('G2_dict.npy', history_dict)
		# calculate mean and std now
		# there r mean and std corresponding to t
		mean_infected_dict = defaultdict(float)
		std_infected_dict = defaultdict(float)
		for time in history_dict:
			mean = statistics.mean(history_dict[time].values())
			std = statistics.pstdev(history_dict[time].values())
			mean_infected_dict[time] = mean
			std_infected_dict[time] = std
		np.save('G2_mean.npy', mean_infected_dict) 
		np.save('G2_std.npy', std_infected_dict)
		print('We have mean_infected_dict and std_infected_dict')
		# print(mean_infected_dict,std_infected_dict)
		return mean_infected_dict, std_infected_dict

	def plot(self):
		mean = np.load('G2_mean.npy')
		std = np.load('G2_std.npy')
		mean_dict = mean.item()
		std_dict = std.item()
		# for key in mean.item():
		# 	print(key, mean.item().get(key))
		x = range(1,int(list(self.Data_Highschool.t)[-1])+1)
		fig, axes = plt.subplots(nrows=1, ncols=1)
		axes.plot(x,mean_dict.values(),color = "blue",linewidth =2)
		axes.errorbar(x, mean_dict.values(),yerr=std_dict.values(),color = "grey")
		axes.set_xlabel("Step(t)")
		axes.set_title("Average number of infected nodes and its variance")
		axes.legend(loc=4)
		plt.show()
		return
def rec_dd():
	return defaultdict(rec_dd)

# print(mean)
# print(type(mean))



if __name__ == '__main__':
	results = network()
	def rec_dd():
		return defaultdict(rec_dd)
