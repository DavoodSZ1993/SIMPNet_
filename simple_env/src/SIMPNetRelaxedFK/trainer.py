#!/usr/bin/env python3

from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch 
import numpy as np 
import os 
import time 
import random
import math

# Custom Modules
from utils import set_random_seed, custom_collate_fn

class Trainer():
    def __init__(self, dataset, model=None, lr=None, num_epochs=None, batch_size=None, weight_decay=None, gaussian_normalization=None):
        '''
        Constructing the trainer class
        '''

        self.model = model 
        self.dataset = dataset 
        self.lr = lr 
        self.num_epochs = num_epochs 
        self.batch_size = batch_size
        self.gaussian_normalization = gaussian_normalization
        self.num_train = None 
        self.num_val = None

        self.train_loader, self.val_loader = self.data_split()
        print(f'Number of training samples: {self.num_train}')
        print(f'Number of validation samples: {self.num_val}')

        # Training Metrics
        self.train_loss = None 
        self.val_loss = None 

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

    def data_split(self):
        '''
        Splitting dataset into train and validation. 80% for training, and 20% for validation.
        '''
        set_random_seed(42)                                    # For reproducibility
        num_items = len(self.dataset)
        indices = list(range(0, len(self.dataset)))
        train_ratio = 0.9
        split = int(math.floor(train_ratio * num_items))

        shuffled_indices = torch.randperm(num_items).tolist()

        train_indices, val_indices = shuffled_indices[:split], shuffled_indices[split:]

        x_train = Subset(self.dataset, train_indices)
        self.num_train = len(x_train)

        x_val = Subset(self.dataset, val_indices)
        self.num_val = len(x_val)

        train_loader = DataLoader(x_train, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(x_val, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)

        return train_loader, val_loader

    def batch_train(self):
        '''
        Training of each batch
        '''

        avg_loss = 0
        self.model.train()
        for graph_batch, obstacle_batch in self.train_loader:
            self.optimizer.zero_grad()
            y_pred = self.model(graph_batch, obstacle_batch)
            loss = self.loss_fn(graph_batch.y, y_pred)
            avg_loss = avg_loss + loss.data.item()
            loss.backward()
            self.optimizer.step()

        return avg_loss

    def batch_eval(self):
        '''
        Evaluating the result
        '''
        eval_loss = 0
        self.model.eval()
        with torch.no_grad():
            for graph_batch, obstacle_batch in self.val_loader:
                y_pred = self.model(graph_batch, obstacle_batch)
                loss = self.loss_fn(graph_batch.y, y_pred)
                eval_loss = eval_loss + loss.data.item()
        
        return eval_loss
                

    def train(self):
        '''
        Train the model for the given epochs
        '''
        
        for epoch in range(1, self.num_epochs + 1):
            avg_loss = self.batch_train()
            avg_loss = avg_loss / (self.num_train / self.batch_size)
            print(f'Epoch: {epoch}, Loss is: {avg_loss}')

            
            eval_loss = self.batch_eval()
            eval_loss = eval_loss / (self.num_val / self.batch_size)
            #print(f'Epoch: {epoch}, Evaluation loss is: {eval_loss}')

        self.train_loss = avg_loss 
        self.val_loss = eval_loss
        print('Training is complete. Now save the model.')
        self.save_model()

    def save_model(self):
        '''
        Saves the trained GNN
        '''

        path = '/home/davood/catkin_ws/src/GNN2/src/trained_models'
        if self.gaussian_normalization:
            file_name = 'gaussian_trained_mainMPNN1.pt'
        else:
            file_name = 'minmax_trained_mainMPNN1.pt' 
              
        timestr = time.strftime("%m-%d-%H%M%S")
        torch.save({
            "epoch": self.num_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": self.train_loss,
            "eval_loss": self.val_loss}, os.path.join(path, file_name))