import torch
import numpy as np
import pandas as pd
import torch.optim as optim

from agents.common.logger import Logger
from agents.common.replay_buffer import ReplayBuffer
from agents.common.utils import quantile_huber_loss

class DropoutAgent():

    def __init__(self,env,
    network,
    dropout=0.1, 
    std_prior=0.01,
    logging=True,
    train_freq=10,
    updates_per_train=100,
    weight_decay=1e-5,
    batch_size=32,
    start_train_step=10,
    log_folder_details=None,
    learning_rate=1e-3,
    verbose=False
    ):

        self.env = env
        self.network = network(env.n_features, std_prior, dropout=dropout)
        self.logging = logging
        self.replay_buffer = ReplayBuffer()
        self.batch_size = batch_size
        self.log_folder_details = log_folder_details
        self.train_freq = train_freq
        self.start_train_step = start_train_step
        self.updates_per_train = updates_per_train
        self.verbose = verbose
        self.dropout = dropout
        self.weight_decay = weight_decay
        
        self.n_samples = 0
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-8, weight_decay=self.weight_decay)


        self.train_parameters = {
        'dropout':dropout,
        'weight_decay':weight_decay,
        'std_prior':std_prior,
        'train_freq':train_freq,
        'updates_per_train':updates_per_train,
        'batch_size':batch_size,
        'start_train_step':start_train_step,
        'learning_rate':learning_rate
        }

    def learn(self,n_steps):

        self.train_parameters['n_steps']=n_steps

        if self.logging:
            self.logger = Logger(self.log_folder_details,self.train_parameters)

        cumulative_regret = 0

        for timestep in range(n_steps):

            self.dropout = 1000*self.dropout/(self.n_samples+1000)

            x = self.env.sample()

            action = self.act(x.float())

            reward = self.env.hit(action)
            regret = self.env.regret(action)

            cumulative_regret += regret

            action = torch.as_tensor([action], dtype=torch.long)
            reward = torch.as_tensor([reward], dtype=torch.float)

            if action ==1:
                self.n_samples += 1
                self.replay_buffer.add(x, reward)

            if self.logging:
                self.logger.add_scalar('Cumulative_Regret', cumulative_regret, timestep)
                self.logger.add_scalar('Mushrooms_Eaten', self.n_samples, timestep)

            if timestep % self.train_freq == 0 and self.n_samples > self.start_train_step:

                if self.verbose:
                    print('Timestep: {}, cumulative regret {}'.format(str(timestep),str(cumulative_regret)))

                for update in range(self.updates_per_train):

                    samples = self.replay_buffer.sample(np.min([self.n_samples,self.batch_size]))
                    self.train_step(samples)

        if self.logging:
            self.logger.save()

    def train_step(self,samples):

        states, rewards = samples

        # Calcul de la TD Target
        target = rewards

        # Calcul de la Q value en fonction de l'action jouée

        q_value = self.network(states.float(),self.dropout)

        loss_function = torch.nn.MSELoss()
        loss = loss_function(q_value.squeeze(), target.squeeze())

        # Update des poids
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def act(self,x):

        action = self.predict(x)

        return action

    @torch.no_grad()
    def predict(self,x):

        estimated_value = self.network(x,self.dropout)

        if estimated_value > 0:
            action = 1
        else:
            action = 0

        return action