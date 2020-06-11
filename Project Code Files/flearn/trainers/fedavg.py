import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import time

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(
            params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Proximal'''
        print(
            'Training with {} workers ---'.format(self.original_params['batch_size']))

        time_log_high = []
        time_log_low = []
        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats = self.test()  # have set the latest model for all clients
                stats_train = self.train_error_and_loss()

                tqdm.write('At round {} accuracy: {}'.format(
                    i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(
                    i, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(
                    i, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])))

            indices_high, selected_clients_high, indices_low, selected_clients_low = self.select_clients(
                i)  # uniform sampling
            np.random.seed(i)
            csolns_high = []  # buffer for receiving client solutions
            csolns_low = []

            round_time_high = 0
            round_time_low = 0

            for idx, c in enumerate(selected_clients_high):
                # communicate the latest model
                c.set_params(self.latest_model_high)

                # solve minimization locally
                startTime = time.time()
                soln, stats = c.solve_inner(
                    num_epochs=self.num_epochs, batch_size=self.batch_size)
                round_time_high += time.time() - startTime

                # gather solutions from client
                csolns_high.append(soln)

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            for idx, c in enumerate(selected_clients_low):
                # communicate the latest model
                # c.set_params((self.latest_model_high[0][c.start:c.start + int(
                #     self.original_params['num_features']/2), :], self.latest_model_high[1]))
                subset_model = ([self.latest_model_high[0][i] for i in c.start], self.latest_model_high[1])
                c.set_params(subset_model)

                # solve minimization locally
                startTime = time.time()
                soln, stats = c.solve_inner(
                    num_epochs=self.num_epochs, batch_size=self.batch_size)
                round_time_low += time.time() - startTime

                t = []
                for i in range(0, len(csolns_high)):
                    t.append(csolns_high[i][1][0])
                t = np.mean(t, axis=0)
                # t[c.start:c.start +
                #     int(self.original_params['num_features']/2)] = soln[1][0]
                j = 0
                for i in range(0, len(t)):
                    if i in c.start:
                        t[i] = soln[1][0][j]
                        j = j + 1
                soln[1][0] = t

                # gather solutions from client
                csolns_low.append(soln)

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            time_log_high.append(round_time_high / len(selected_clients_high))
            time_log_low.append(round_time_low / len(selected_clients_low))

            temp = []
            for i in range(0, (len(selected_clients_high) + len(selected_clients_low))):
                if i < len(selected_clients_high):
                    temp.append(csolns_high[i])
                else:
                    temp.append(csolns_low[i-len(selected_clients_high)])

            temp = np.asarray(temp)
            self.latest_model_high = self.aggregate(temp)

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(
            self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(
            self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))

        print('Timings:\n High device avg:', np.mean(time_log_high))
        print('Timings:\n Low device avg:', np.mean(time_log_low))
        tqdm.write(' '.join(str(round(e,4)) for e in time_log_high))
        tqdm.write('\n')
        tqdm.write(' '.join(str(round(e,4)) for e in time_log_low))
