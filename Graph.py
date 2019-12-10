import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import linear_sum_assignment
import pandas as pd

# Initializing the graph object
class Graph:
    def __init__(self, edges_list):
        self.list_edges = edges_list
        self.list_vertices = self.get_vertices()
        self.nb_vertices = len(self.list_vertices)

    def get_vertices(self):
        liste_unique = list()
        for i in self.list_edges:
            for j in i:
                if j not in liste_unique:
                    liste_unique.append(j)
        liste_unique.sort()
        return liste_unique

    def get_parents(self, node):
        res = []
        for i in self.edges:
            if i[1]==node:
                res.append(i[0])
        return res

# function used for zero padding of two vectors having different lengths
    def pad(self,vec1,vec2):

        if len(vec1)>len(vec2):
            vec2 = np.r_[vec2,np.zeros((len(vec1)-len(vec2)))]
        else:
            vec1 = np.r_[vec1,np.zeros((len(vec2)-len(vec1)))]
        return vec1,vec2

# function used to find the degree of a given vertex in the graph object
    def get_vertex_degree(self, i):
        degree = 0
        for j in self.list_edges:
            if i in j:
                degree+=1
        return degree

# function calculating the prob vector representation of the graph
    def get_prob_vector(self):
        vector = np.zeros((self.nb_vertices))
        for i in range(1,len(vector)+1):
            for j in self.list_vertices:
                if self.get_vertex_degree(j)==i:
                    vector[i-1]+=1
        return vector

# function printing a visual representation of the graph
    def print_graph(self):
        G = nx.DiGraph()
        G.add_edges_from(self.list_edges)
        pos = nx.circular_layout(G)
        G = G.to_undirected()
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edge_color='k', width=0.5)
        return 0

# function calculating the prob_distance from a given graph
    def get_prob_distance(self, graph):
        prob1 = self.get_prob_vector()
        prob2 = graph.get_prob_vector()
        prob1, prob2 = self.pad(prob1,prob2)
        return np.linalg.norm(prob1 - prob2,1)

#
    def get_adjacent_vertices(self,i):
        res = list()
        for j in self.list_edges:
            if i in j:
                for k in j:
                    if k!=i:
                        res.append(k)
        return res

    def get_node_signature(self,i):
        res = list()
        res.append(self.get_vertex_degree(i))
        for j in self.get_adjacent_vertices(i):
            res.append(self.get_vertex_degree(j))
        return res

    def get_vector_signatures(self):
        res = list()
        for i in self.list_vertices:
            res.append(self.get_node_signature(i))
        return res

    # nb_lines = len(self.get_vector_signature())
    # nb_col = len(graph.get_vector_signature())
    # cost_matrix[i,j] = L1(self.get_node_signature(i),graph.get_node_signature(j))
    def build_cost_matrix(self, graph):
        cost_matrix = np.zeros((self.nb_vertices,graph.nb_vertices))
        l1 = self.get_vector_signatures()
        l2 = graph.get_vector_signatures()
        for i in range(cost_matrix.shape[0]):
            vec1 = l1[i]
            for j in range(cost_matrix.shape[1]):
                vec2 = l2[j]
                vec1, vec2 = self.pad(vec1,vec2)
                cost_matrix[i,j]=np.linalg.norm(vec1 - vec2,1)
        return cost_matrix

    def get_matching_and_cost(self, graph):
        cost_matrix = self.build_cost_matrix(graph)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        M = len(row_ind)
        cost = cost_matrix[row_ind,col_ind].sum()
        return cost, M


    def get_matching_distance(self, graph):
        cost, M = self.get_matching_and_cost(graph)
        if M==0:
            M=1
        return ((cost/M) + np.abs(self.nb_vertices-graph.nb_vertices))
