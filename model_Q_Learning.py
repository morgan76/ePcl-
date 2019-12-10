import numpy as np
import pandas as pd
from collections import deque
import pickle
from scipy import sparse

class Q_Learning(object):

# Initializing the Q learning object
    def __init__(self, adjacency_file_name, Q_matrix, gamma, alpha, MEMORY_SIZE, WINDOW_SIZE,sitemap):
        self.adjacency_matrix = np.array(pd.read_csv(str(adjacency_file_name)+'.csv').iloc[:,1:])
        self.pages_values = np.arange(0,65,1)
        self.nb_pages = len(self.pages_values)
        self.liste_actions  = list(self.pages_values)
        states_list = list()
        for i in range(len(self.pages_values)):
            temp = list()
            temp.append(self.pages_values[i])
            states_list.append(temp)
            for j in range(len(self.pages_values)):
                temp = list()
                temp.append(self.pages_values[i])
                temp.append(self.pages_values[j])
                states_list.append(temp)
                for k in range(len(self.pages_values)):
                    temp = list()
                    temp.append(self.pages_values[i])
                    temp.append(self.pages_values[j])
                    temp.append(self.pages_values[k])
                    states_list.append(temp)
        self.states_list = states_list
        self.nb_states = len(states_list)
        try:
            self.Q = sparse.load_npz(str(Q_matrix)+".npz").toarray()
        except FileNotFoundError:
            self.Q = np.zeros((self.nb_states, self.nb_pages))
        self.V = np.zeros([self.nb_states,self.nb_pages])
        self.gamma = gamma
        self.alpha = alpha
        self.WINDOW_SIZE = WINDOW_SIZE
        self.predictions_list = deque(maxlen=MEMORY_SIZE)
        self.online_episode = list()
        self.online_times = list()
        self.sitemap_usage = sitemap
        self.compteur = 0
        self.history = deque(maxlen=MEMORY_SIZE)
        self.liste_rewards = []

# function used to convert a state to its index in the Q matrix
    def convert_s_to_i(self,state,states_list):
        return states_list.index(list(state))

# transition function from one state to another given a specific action
    def delta(self,state, action):
        res = state.copy()
        res.append(action)
        return res

# function selecting the best action following an epsilon-greedy policy
    def select_next_action(self,current_state):
        last_page = current_state[-1]
        epsilon = np.random.randint(0, 10)

        if epsilon>1:
            # selcting the best action based on the Q matrix
            if self.sitemap_usage:
                next_action = np.argmax(np.multiply(self.Q[self.convert_s_to_i(current_state,self.states_list),:],self.adjacency_matrix[last_page,:]))
                next_state = self.delta(current_state,next_action)
            else:
                next_action = np.argmax(self.Q[self.convert_s_to_i(current_state,self.states_list),:])
                next_state = self.delta(current_state,next_action)
            return next_action, next_state
        else:
            # selecting a random action
            res = np.random.randint(0, self.nb_pages)
            while self.adjacency_matrix[last_page,res]==0:
                res = np.random.randint(0, self.nb_pages)
            next_state = self.delta(current_state,res)
            return res, next_state

# function updating the list of predictions made by the agent
    def update_pred_list(self,new_value):
        self.predictions_list.append(new_value)
        return

# function calculating the reward based on the action selected and the real next page
    def calculate_reward(self, answer, time_spent):
        reward = 0
        index = 100000
        if answer[-1] in self.predictions_list:
            index = self.predictions_list.index(answer[-1])
            reward = self.alpha*(len(self.predictions_list)-index)+(1-self.alpha)*time_spent[0]/5000
        return reward, index

# function updating the Q matrix
    def update_Q(self, current_state, next_action, reward):
        index_current = self.convert_s_to_i(current_state, self.states_list)
        index_predicted = self.convert_s_to_i(self.delta(current_state, next_action),self.states_list)
        self.V[index_current,next_action]+=1
        v = 1/(self.V[index_current,next_action])
        self.Q[index_current,next_action] = (1-v)*self.Q[index_current,next_action]+v*(reward+self.gamma*np.max(self.Q[index_predicted,:]))
        return

# function managing the training of the agent (offline use)
    def train(self, episode, times_spent):
        current_state = deque(maxlen=self.WINDOW_SIZE)
        i = 0

        while i<len(episode)-1:
            current_state.append(episode[i])
            next_action, predicted_state = self.select_next_action(current_state)
            answer = self.delta(current_state,episode[i+1])
            self.update_pred_list(predicted_state[-1])
            self.history.append([self.convert_s_to_i(current_state,self.states_list),predicted_state[-1]])
            reward, index_good_answer = self.calculate_reward(answer,times_spent[i+1])
            self.liste_rewards.append(reward)
            self.update_Q(current_state, next_action, reward)
            i+=1
        self.compteur+=1
        if self.compteur%1000==0:
            Q_storage = sparse.csr_matrix(self.Q)
            sparse.save_npz("Q_storage.npz",Q_storage)
        return

# function giving the agent's prediction (online use)
    def predict(self, page_number, time_spent):
        self.online_episode.append(page_number)
        self.online_times.append(time_spent)
        current_state = self.online_episode[-self.WINDOW_SIZE:]
        index = np.argmax(self.Q[self.convert_s_to_i(current_state,self.states_list),:])
        array = self.Q[self.convert_s_to_i(current_state,self.states_list),:]
        if np.max(array)==0:
            temp_state = current_state[-1]
            array = self.Q[self.convert_s_to_i(temp_state,self.states_list),:]
        if np.max(array)==0:
            temp_state = current_state[-1]
            array = np.array(['?'])
            return 0
        return (-array).argsort()[0]

# function to empty the last session witnessed by the agent
    def reset(self):
        self.online_episode = list()
        self.online_times = list()
        return


    def save(self, fileName):
        """Save thing to a file."""
        pickle.dump(self,open('model.pkl','wb'))
