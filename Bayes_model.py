import numpy as np
import pandas as pd
import Graph
import pickle
import itertools
from sklearn.metrics import mutual_info_score


# Creating the bayes net object
class Bayes_net(object):



    def __init__(self,filename,beta,naive,filename_fit_tables,filename_indexes_tables):
        # Load saved data
        self.filename = filename
        try:
            self.data = pickle.load(open(str(self.filename)+'.pkl','rb'))
            print('Data found!')
        except FileNotFoundError:
            self.data = {}
        # Regularisation
        self.beta = beta
        # Vertices of the graph built by Chow-Liu algorithm
        self.edges = list()
        self.graph = {}
        # List of unique values for each variable, used to build contingency tables later
        self.list_omegas = list()
        self.compteur = 0
        # Set the naive boolean to true or false, used to determine the architecture of the network
        self.naive = naive
        # Load tables and indexes if already built
        self.filename_fit_tables= filename_fit_tables
        self.filename_indexes_tables = filename_indexes_tables
        try:
            self.fit_tables =  pickle.load(open(str(self.filename_fit_tables)+'.pkl','rb'))
            print('Fit tables found!')
        except FileNotFoundError:
            self.fit_tables = []
        try:
            self.indexes_tables =  pickle.load(open(str(self.filename_indexes_tables)+'.pkl','rb'))
            print('Indexes tables found!')
        except FileNotFoundError:
            self.indexes_tables = []

# function building the list of omegas for all the variables
    def build_list_of_omegas(self):
        print('building omegas')
        for i in range(len(list(self.data.items())[0][1])):
            print(i)
            self.list_omegas.append(np.unique([list(self.data.items())[k][1][i] for k in range(len(list(self.data.items())))]).tolist())
        return

# function finding an element in an omega list based on its value
    def find(self, value, index):
        answer = list()
        for i in range(len(list(self.data.items()))):
            if list(self.data.items())[i][1][index]==value:
                answer.append(i)
        return answer

# function building a table for the group of variables having indexes contained in liste_indexes
# generates all the possible combinations between the different values of the variables, and then
# counts how many times  in our data this combination occurs
    def build_table(self, list_indexes):
        list_of_lists = [self.list_omegas[i] for i in list_indexes]
        combinations = list(itertools.product(*list_of_lists))
        resp = []
        final = []
        r_final = []
        for i in combinations:
            temp = []
            temp.append(i)
            temp.append(0)
            r_final.append(temp)
        for i in range(len(combinations)):
            temp = []
            temp.append(combinations[i])
            temp_2 = []
            for j in range(len(list_indexes)):
                temp_2.append(self.find(combinations[i][j],list_indexes[j]))
            temp.append(temp_2)
            resp.append(temp)
            min_length=1000000000000000000000000
            index = -7
            temp = []
            for j in range(len(resp[i][1])):
                if len(resp[i][1][j])<min_length:
                    min_length=len(resp[i][1][j])
                    index=j
            temp = resp[i][1][index]
            k=len(temp)-1
            res = len(temp)
            for k in range(len(temp)):
                bool = True
                for j in range(len(resp[i][1])):
                    if temp[k] not in resp[i][1][j]:
                        bool= False
                if bool ==False:
                    res-=1
            r_final[i][1]=res
        return r_final

# function updating tables contained in self.fit_tables, looks if the new value exists already in each list
    def update_tables(self, values):
        for i in range(len(self.indexes_tables)):
            key = ()
            for j in self.indexes_tables[i]:
                key = key + (values[j],)
            values_=[]
            for j in self.indexes_tables[i]:
                values_.append(values[j])
            # the value doe not exist
            if self.find_index_in_table(self.fit_tables[i],values_)==-1:
                self.fit_tables[i].append([key,1])
            else:
            # the value does exist, we add one
                index = self.find_index_in_table(self.fit_tables[i],values_)
                self.fit_tables[i][index][1]+=1
        return

# function adding new users, should be used the first time the network is created
    def add_user(self,  user_id, nb_sessions, nb_produits, nb_categories, min_time, max_time, avg_time, subscription ):
        temp = []
        temp.append(nb_sessions)
        temp.append(nb_produits)
        temp.append(nb_categories)
        temp.append(self.process_time(min_time))
        temp.append(self.process_time(max_time))
        temp.append(self.process_time(avg_time))
        temp.append(subscription)
        self.data[user_id]=temp
        self.compteur+=1
        if self.compteur%500==0:
            pickle.dump(self.data,open(str(self.filename)+'.pkl','wb'))
        return

# function adding new users, should be used once the network is created
    def add_new_user(self, user_id, nb_sessions, nb_produits, nb_categories, min_time, max_time, avg_time, subscription):
        temp = []
        temp.append(nb_sessions)
        temp.append(nb_produits)
        temp.append(nb_categories)
        temp.append(self.process_time(min_time))
        temp.append(self.process_time(max_time))
        temp.append(self.process_time(avg_time))
        temp.append(subscription)
        self.data[user_id]=temp
        self.compteur+=1
        pickle.dump(self.data,open(str(self.filename)+'.pkl','wb'))
        self.update_tables(temp)
        return

# function calculating the mututal information between two variables using their indexes
    def calculate_MI(self, var_index1, var_index2):
        list_1 = list()
        list_2 = list()
        for i in self.data.keys():
            list_1.append(self.data[i][var_index1])
            list_2.append(self.data[i][var_index2])
        return mutual_info_score(list_1,list_2)

# function calculating the matrix of mutual informations
    def build_MI_matrix(self,size):
        matrix_adjacence_MI_graph = np.zeros((size,size))
        print('Building Matrix')
        for i in range(len(matrix_adjacence_MI_graph)):
            for j in range(i):
                if i!=j:
                    value = self.calculate_MI(i,j)
                    matrix_adjacence_MI_graph[i,j]=value
                    matrix_adjacence_MI_graph[j,i]=value
        return matrix_adjacence_MI_graph

# Prim's algorithm to determine the maximum spanning tree in a graph, the links between variables are stored in self.edges
# must be called without the variable 6 !!!
    def prim(self,size):
        vertices = list()
        vertices.append(0)
        m_adj = self.build_MI_matrix(size)
        while(len(vertices))<size:
            max=-1000000
            ind_max=1000
            for i in range(len(vertices)):
                for j in range(len(m_adj)):
                    if m_adj[vertices[i],j]>max and j not in vertices:
                        max = m_adj[vertices[i],j]
                        ind_max = j
                        ind_dep = vertices[i]
            vertices.append(ind_max)
            self.edges.append([ind_dep,ind_max])
        return

# function giving orientation to self.edges (not necessary)
    def give_orientation(self):
        edges_copy = self.edges.copy()
        for i in range(len(self.edges)):
            self.graph[i]=[]
        visited = []
        visited.append(len(self.edges))
        for l in visited:
            for k in range(len(edges_copy)):
                if l in edges_copy[k]:
                    for j in edges_copy[k]:
                        if j!=l and j not in visited:
                            if l not in self.graph[j]:
                                self.graph[j].append(l)
                            visited.append(j)
        return

# function returning the probability in a table for a certain list of values
    def find_prob_in_table(self,d, values):
            res = 0
            found = True
            key = ()
            for i in values:
                key = key + (i,)
            for i in d:
                if i[0]==key:
                    return i[1]
            return res

# function returning the index in a table of a certain list of values
    def find_index_in_table(self,d,values):
            res = 0
            found = True
            key = ()
            for i in values:
                key = key + (i,)
            for i in range(len(d)):
                if d[i][0]==key:
                    return i
            return -1

# function converting time into categories
    def process_time(self,time):
        for i in range(160):
            if time>160_000-1000*i:
                return 160_000-1000*i

# function associating for each variable its parents in the graph, if naive, every variable's parent is variable 6 (subscription)
# if not naive, depends on self.edges --> calls self.prim(6)
    def merge_edges(self):
        self.build_list_of_omegas()
        tables = []
        if self.naive == False:
            self.prim(6)
            for i in range(len(np.unique(self.edges))):
                tables.append([])
            for i in range(len(self.edges)):
                for j in range(len(tables)):
                    if j==self.edges[i][1]:
                        tables[j].append(self.edges[i][0])
            for j in range(len(tables)):
                tables[j].append(6)
        else:
            for i in range(6):
                tables.append([6])
            print('Tables',tables)
        return tables

# fits the model to its data
# 1) Finds the right architecture
# 2) Calculates the right tables and saves them in files for later
    def fit(self):
        if len(self.fit_tables)==0:
            merged_edges = self.merge_edges()
            print('Merged_edges',merged_edges)
            for i in range(len(merged_edges)):
                temp = []
                for j in range(len(merged_edges[i])):
                    temp.append(merged_edges[i][j])
                temp.append(i)
                print('adding table ',temp)
                self.fit_tables.append(self.build_table(temp))
                self.indexes_tables.append(temp)
                print('adding table ',temp[:-1])
                self.fit_tables.append(self.build_table(temp[:-1]))
                self.indexes_tables.append(temp[:-1])
            pickle.dump(self.fit_tables,open(str(self.filename_fit_tables)+'.pkl','wb'))
            pickle.dump(self.indexes_tables,open(str(self.filename_indexes_tables)+'.pkl','wb'))
        return

# function giving P(sub=True|.....)
    def give_num(self,known_values):
        temp = 1
        i=0
        while i<len(self.fit_tables)-1:
            liste_indexes = []
            liste_indexes_2 = []
            for k in self.indexes_tables[i]:
                liste_indexes.append(known_values[k])
            for k in self.indexes_tables[i+1]:
                liste_indexes_2.append(known_values[k])
            temp*=(self.find_prob_in_table(self.fit_tables[i],liste_indexes)+1)/(self.find_prob_in_table(self.fit_tables[i+1],liste_indexes_2)+len(self.fit_tables[i]))
            i+=2
        return temp

# function giving P(sub=True|.....)/(P(sub=True|.....)+P(sub=False|.....))
    def predict(self,nb_sessions,nb_prod,nb_cat,min_time,max_time,avg_time,sub,prob_desired,prob_else):
        temp_1 = self.give_num([nb_sessions,nb_prod,nb_cat,min_time,max_time,avg_time,sub])
        temp_2 = self.give_num([nb_sessions,nb_prod,nb_cat,min_time,max_time,avg_time,1-sub])
        temp_1 *= prob_desired
        temp_2 *= prob_else
        return temp_1/(temp_2+temp_1)
