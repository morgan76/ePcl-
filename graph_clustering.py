import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import linear_sum_assignment
from Graph import *
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from progressbar import ProgressBar
from scipy import sparse
import pickle
import os.path


class Graph_clustering(object):

# function used for zero padding of two vectors having different lengths
    def pad(vec1,vec2):

        if len(vec1)>len(vec2):
            vec2 = np.r_[vec2,np.zeros((len(vec1)-len(vec2)))]
        else:
            vec1 = np.r_[vec1,np.zeros((len(vec2)-len(vec1)))]
        return vec1,vec2

# function converting a path to a graph object
    def convert_path_to_graph(self,path):
        graph = []
        for i in range(len(path)-1):
            if isinstance(path[i], list):
                if int(path[i][0])!=int(path[i+1][0]):
                    graph.append([int(path[i][0]),int(path[i+1][0])])
            elif int(path[i])!=int(path[i+1]):
                graph.append([int(path[i]),int(path[i+1])])
        return Graph(graph)


# function processing raw data (optional)
    # def process_data(self,filename):
    #     data = pd.read_csv(str(filename)+'.csv')
    #     categories = {
    #             'http://www.cdiscount.com/home.html':0,
    #             'http://www.cdiscount.com/cat_1':1,
    #             'http://www.cdiscount.com/cat_2':2,
    #             'http://www.cdiscount.com/cat_3':3,
    #             'http://www.cdiscount.com/cat_4':4,
    #             'http://www.cdiscount.com/cat_5':5,
    #             'http://www.cdiscount.com/cat_6':6,
    #             'http://www.cdiscount.com/cat_7':7,
    #             'http://www.cdiscount.com/cat_8':8,
    #             'http://www.cdiscount.com/cat_9':9,
    #             'http://www.cdiscount.com/cat_10':10,
    #             'http://www.cdiscount.com/FAQ':11,
    #             'http://www.cdiscount.com/monpannier':12,
    #             'http://www.cdiscount.com/cat_1/produit_1':13,
    #             'http://www.cdiscount.com/cat_1/produit_2':14,
    #             'http://www.cdiscount.com/cat_1/produit_3':15,
    #             'http://www.cdiscount.com/cat_1/produit_4':16,
    #             'http://www.cdiscount.com/cat_1/produit_5':17,
    #
    #             'http://www.cdiscount.com/cat_2/produit_1':18,
    #             'http://www.cdiscount.com/cat_2/produit_2':19,
    #             'http://www.cdiscount.com/cat_2/produit_3':20,
    #             'http://www.cdiscount.com/cat_2/produit_4':21,
    #             'http://www.cdiscount.com/cat_2/produit_5':22,
    #
    #             'http://www.cdiscount.com/cat_3/produit_1':23,
    #             'http://www.cdiscount.com/cat_3/produit_2':24,
    #             'http://www.cdiscount.com/cat_3/produit_3':25,
    #             'http://www.cdiscount.com/cat_3/produit_4':26,
    #             'http://www.cdiscount.com/cat_3/produit_5':27,
    #
    #             'http://www.cdiscount.com/cat_4/produit_1':28,
    #             'http://www.cdiscount.com/cat_4/produit_2':29,
    #             'http://www.cdiscount.com/cat_4/produit_3':30,
    #             'http://www.cdiscount.com/cat_4/produit_4':31,
    #             'http://www.cdiscount.com/cat_4/produit_5':32,
    #
    #             'http://www.cdiscount.com/cat_5/produit_1':33,
    #             'http://www.cdiscount.com/cat_5/produit_2':34,
    #             'http://www.cdiscount.com/cat_5/produit_3':35,
    #             'http://www.cdiscount.com/cat_5/produit_4':36,
    #             'http://www.cdiscount.com/cat_5/produit_5':37,
    #
    #             'http://www.cdiscount.com/cat_6/produit_1':38,
    #             'http://www.cdiscount.com/cat_6/produit_2':39,
    #             'http://www.cdiscount.com/cat_6/produit_3':40,
    #             'http://www.cdiscount.com/cat_6/produit_4':41,
    #             'http://www.cdiscount.com/cat_6/produit_5':42,
    #
    #             'http://www.cdiscount.com/cat_7/produit_1':43,
    #             'http://www.cdiscount.com/cat_7/produit_2':44,
    #             'http://www.cdiscount.com/cat_7/produit_3':45,
    #             'http://www.cdiscount.com/cat_7/produit_4':46,
    #             'http://www.cdiscount.com/cat_7/produit_5':47,
    #
    #             'http://www.cdiscount.com/cat_8/produit_1':48,
    #             'http://www.cdiscount.com/cat_8/produit_2':49,
    #             'http://www.cdiscount.com/cat_8/produit_3':50,
    #             'http://www.cdiscount.com/cat_8/produit_4':51,
    #             'http://www.cdiscount.com/cat_8/produit_5':52,
    #
    #             'http://www.cdiscount.com/cat_9/produit_1':53,
    #             'http://www.cdiscount.com/cat_9/produit_2':54,
    #             'http://www.cdiscount.com/cat_9/produit_3':55,
    #             'http://www.cdiscount.com/cat_9/produit_4':56,
    #             'http://www.cdiscount.com/cat_9/produit_5':57,
    #
    #             'http://www.cdiscount.com/cat_10/produit_1':58,
    #             'http://www.cdiscount.com/cat_10/produit_2':59,
    #             'http://www.cdiscount.com/cat_10/produit_3':60,
    #             'http://www.cdiscount.com/cat_10/produit_4':61,
    #             'http://www.cdiscount.com/cat_10/produit_5':62,
    #
    #             'http://www.cdiscount.com/livraison':63,
    #             'http://www.cdiscount.com/commande':64,
    #             }
    #     data['urls'] = [categories[item] for item in data['urls']]
    #     #data['urls']=data['urls'].astype("category").cat.codes
    #     liste_users_id = np.unique(data.user_id)
    #     values_pages = np.unique(data['urls'])
    #     nb_pages = len(values_pages)
    #     res = np.zeros((len(liste_users_id)-1,35))
    #     j=0
    #     pbar = ProgressBar()
    #     for i in pbar(range(len(liste_users_id)-1)):
    #         k = 0
    #         while data.user_id[j]==liste_users_id[i]:
    #             res[i,k]=data.urls[j]
    #             j+=1
    #             k+=1
    #     liste_graphs = []
    #     for i in range(len(res)):
    #         liste_graphs.append(self.convert_path_to_graph(res[i]))
    #     return liste_graphs

# Initializing the graph clustering object
    def __init__(self,data_file_name, distance_matrix_file_name, distance_type,n_clusters):
        self.distance_type = distance_type
        try:
            self.distance_matrix = sparse.load_npz(str(distance_matrix_file_name)+".npz").toarray()
        except FileNotFoundError:
            self.distance_matrix = np.array(pd.read_csv(str(distance_matrix_file_name)+'.csv').iloc[:,1:])
        self.agg = AgglomerativeClustering(n_clusters=n_clusters, affinity = 'precomputed', linkage='average')
        if self.distance_type=='prob':
            try :
                with open('liste_graphs_prob', 'rb') as f:
                    self.liste_graphs = pickle.load(f)
                    f.close()
            except FileNotFoundError:
                self.liste_graphs = self.process_data(data_file_name)
        else:
            try :
                with open('liste_graphs_match', 'rb') as f:
                    self.liste_graphs = pickle.load(f)
                    f.close()
            except FileNotFoundError:
                self.liste_graphs = self.process_data(data_file_name)
        self.data_file_name = data_file_name

# fit clustering
    def fit(self):
        self.agg.fit(self.distance_matrix)
        return

# function returning the clustering labels
    def labels_(self):
        return self.agg.labels_

# function returning the clustering centroids
    def get_centroid_index(self, i):
        distances_sum = list()
        indexes = np.where(self.agg.labels_==i)[0]
        indexes_total = np.zeros((len(self.distance_matrix)))
        sum = 0
        for j in range(len(indexes_total)):
            if j in indexes:
                sum = 0
                for k in indexes:
                    sum += self.distance_matrix[j,k]
            indexes_total[j]=sum
        # centroids are the individuals minimizing the intra-cluster sum of distances
        centroid_index = np.argmin(indexes_total)
        return centroid_index

# function returning the indexes of the centroids in the graphs list
    def get_centroids_indexes(self):
        centroids_indexes = list()
        for i in range(self.agg.n_clusters):
            centroids_indexes.append(self.get_centroid_index(i))
        return centroids_indexes

# function returning the nearest centroid for a given graph, used to assign a new graph to a cluster
    def find_nearest_centroid(self, graph):
        list_distances = list()
        if self.distance_type=='prob':
            for i in self.get_centroids_indexes():
                list_distances.append(graph.get_prob_distance(self.liste_graphs[i]))
        else:
            for i in self.get_centroids_indexes():
                list_distances.append(graph.get_matching_distance(self.liste_graphs[i]))
        print(list_distances)
        res = np.argmin(list_distances)
        return res

# function adding a new graph to the existing data and upating the distance matrix
    def add_new_graph(self, path):
        graph = self.convert_path_to_graph(path)
        if graph not in self.liste_graphs:
            self.liste_graphs.append(graph)
            distances = np.zeros((len(self.liste_graphs)))
            pbar = ProgressBar()
            for i in pbar(range(len(self.liste_graphs)-1)):
                if self.distance_type=='prob':
                    distances[i]=graph.get_prob_distance(self.liste_graphs[i])
                else:
                    distances[i]=graph.get_matching_distance(self.liste_graphs[i])
            temp = np.zeros((len(self.liste_graphs),len(self.liste_graphs)))
            temp[:len(self.liste_graphs)-1,:len(self.liste_graphs)-1] = self.distance_matrix
            temp[-1,:]=distances
            temp[:,-1]=distances
            self.distance_matrix = temp
            if self.distance_type=='prob':
                distance_prob_storage = sparse.csr_matrix(self.distance_matrix)
                sparse.save_npz("distance_prob_storage.npz",distance_prob_storage)
                with open('liste_graphs_prob', 'wb') as f:
                    pickle.dump(self.liste_graphs, f)
            else:
                distance_matching_storage = sparse.csr_matrix(self.distance_matrix)
                sparse.save_npz("distance_matching_storage.npz",distance_matching_storage)
                with open('liste_graphs_match', 'wb') as f:
                    pickle.dump(self.liste_graphs, f)
        return


# function returning the closest cluster for a given graph
    def predict(self,path):
        graph = self.convert_path_to_graph(path)
        answer = self.find_nearest_centroid(graph)
        return answer

# function returning information about each existing cluster : nb_vertices, nb_edges, avg_degree, index of the centroid and the number of individuals in the cluster
    def analyse_cluster(self, i):
        indexes = np.where(self.agg.labels_==i)[0]
        print(len(indexes))
        nb_vertices = list()
        nb_edges = list()
        degrees = list()
        nb_inds = len(indexes)
        pqbar = ProgressBar()
        for j in pqbar(indexes):
            nb_vertices.append(self.liste_graphs[j].nb_vertices)
            nb_edges.append(len(self.liste_graphs[j].list_edges))
            temp = list()
            for k in self.liste_graphs[j].list_vertices:
                temp.append(self.liste_graphs[j].get_vertex_degree(k))
            degrees.append(np.mean(temp))
        return np.mean(nb_vertices), np.median(nb_edges), np.mean(degrees), self.liste_graphs[self.get_centroid_index(i)], nb_inds
