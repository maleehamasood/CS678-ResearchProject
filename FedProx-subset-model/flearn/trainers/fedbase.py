import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad

class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val)
        self.original_params = params
        # create worker nodes
        tf.reset_default_graph()

        random.seed(123)

        self.client_model_low = learner(*params['model_params'], self.inner_opt, self.seed, device_type="low")
        self.client_model_high = learner(*params['model_params'], self.inner_opt, self.seed, device_type="high")
        
        # Based on the drop percentage, some devices will get a smaller version of the model to train.
        self.clients_low, self.clients_high= self.setup_clients(dataset, self.client_model_low, self.client_model_high)
        self.clients = self.clients_low + self.clients_high
        
        print('{} Clients in Total'.format(params['batch_size']))
        print('{} High End Clients, {} Low End Clients'.format(self.original_params['batch_size'] - int(self.original_params['drop_percent'] * self.original_params['batch_size']),int(self.original_params['drop_percent'] * self.original_params['batch_size'])))
        self.latest_model_low = self.client_model_low.get_params()
        self.latest_model_high = self.client_model_high.get_params()

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)

    def __del__(self):
        self.client_model_high.close()
        self.client_model_low.close()

    def setup_clients(self, dataset, model=None, model1=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        client_low = []
        client_high = []
        # print(len(users))
        i = 0
        for u, g in zip(users, groups):
            if i < self.original_params['drop_percent'] * len(users):
                k = 0
                if "synthetic" in self.original_params['dataset']:
                    self.original_params['num_features'] = 60
                    k = random.randint(k,self.original_params['num_features']/2 - 1)
                    for j in range(0, len(train_data[u]['x'])):
                        train_data[u]['x'][j] = train_data[u]['x'][j][k:k+int(self.original_params['num_features']/2) ]
                    for j in range(0, len(test_data[u]['x'])):
                        test_data[u]['x'][j] = test_data[u]['x'][j][k:k+int(self.original_params['num_features']/2)]

                client_low.append(Client(u, g, train_data[u], test_data[u], model, k))
            else:
                client_high.append(Client(u, g, train_data[u], test_data[u], model1))
            i = i + 1

        return client_low, client_high

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses


    def show_grads(self):  
        '''
        Return:
            gradients on all workers and the global gradient
        '''

        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)  

        intermediate_grads = []
        samples=[]

        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model) 
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads.append(client_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples)) 
        intermediate_grads.append(global_grads)

        return intermediate_grads
 
  
    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model_high.set_params(self.latest_model_high)
        # self.client_model_low.set_params(self.latest_model_low)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def save(self):
        pass

    def select_clients(self, round):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        num_clients_high = self.original_params['batch_size'] - int(self.original_params['drop_percent'] * self.original_params['batch_size'])
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        indices_high = np.random.choice(range(len(self.clients_high)), num_clients_high, replace=False)

        num_clients_low = int(self.original_params['batch_size'] - num_clients_high) 
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        indices_low = np.random.choice(range(len(self.clients_low)), num_clients_low, replace=False)

        return indices_high, np.asarray(self.clients_high)[indices_high], indices_low, np.asarray(self.clients_low)[indices_low]

    def aggregate(self, wsolns):
        total_weight = 0.0
        base = [0]*len(wsolns[0][1])
        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w*v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]

        return averaged_soln

