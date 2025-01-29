#!/usr/bin/env python3 

from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch 
import numpy as np 
import os 
import time
import random

class Trainer():
    def __init__(self, model, dataset: list, num_epochs:int, batch_size:int, gaussian_normalization):
        '''
        Constructing the trainer class
        '''

        self.model = model 
        self.dataset = dataset 
        self.batch_size = batch_size 
        self.gaussian_normalization = gaussian_normalization
        self.num_train = None 
        self.num_val = None 

        self.train_loader, self.val_loader = self.data_split()

        self.num_epochs = num_epochs 
        self.writer = SummaryWriter()

        # Training Metrics
        self.train_loss = None
        self.val_loss = None

    def data_split(self)->tuple:
        '''
        Splitting dataset into train, and validation. 80% for training, and 20% for validation.
        '''

        indices = list(range(0, len(self.dataset)))
        #random.shuffle(indices)
        train_ratio = 0.9
        train_data_numbers = int(len(indices) * train_ratio)
        valid_data_numbers = len(indices) - train_data_numbers

        train_data_indices = indices[0: train_data_numbers]
        valid_data_indices = indices[train_data_numbers:]

        x_train = [self.dataset[i] for i in train_data_indices]
        self.num_train = len(x_train)

        x_val = [self.dataset[i] for i in valid_data_indices]
        self.num_val = len(x_val)

        train_loader = DataLoader(x_train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(x_val, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader 

    def batch_train(self)->None:
        '''
        Training of each batch
        '''

        avg_loss = 0
        self.model.train()
        for batch in self.train_loader:
            self.model.optimizer.zero_grad()
            y_pred = self.model(batch)
            y = batch.y.reshape(-1, 12)
            loss = self.model.loss_fn(y_pred, y)
            avg_loss = avg_loss + loss.data.item()
            loss.backward()
            self.model.optimizer.step()

        return avg_loss

    def batch_val(self)->None:
        '''
        Validation for each batch
        '''

        eval_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                y_pred = self.model(batch)
                y = batch.y.reshape(-1, 12)
                eval_loss = eval_loss + self.model.loss_fn(y_pred, y)
        
        return eval_loss

    def train(self)->None:
        '''
        Train the model for the given epochs
        '''

        for epoch in range(1, self.num_epochs + 1):
            avg_loss = self.batch_train()
            avg_loss = avg_loss / (self.num_train / self.batch_size)
            print(f'Epoch: {epoch}, Loss is: {avg_loss}')


            eval_loss = self.batch_val()
            eval_loss = eval_loss / (self.num_val / self.batch_size)
            print(f'Epoch: {epoch}, Evaluation loss is: {eval_loss}')

        self.train_loss = avg_loss 
        self.val_loss = eval_loss 
        print('Training is complete. Now save the model')
        self.save_model()

    def save_model(self)->None:
        '''
        Saves the trained GNN
        '''

        path = '/home/davood/catkin_ws/src/GNN2/src/trained_models'
        if self.gaussian_normalization:
            file_name = 'gaussian_trained_GCN.pt'
        else:
            file_name = 'minmax_trained_GCN.pt'

        # Save other parameters and also save the data
        timestr = time.strftime("%m-%d-%H%M%S")
        torch.save({
            "epoch": self.num_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.model.optimizer.state_dict(),
            "train_loss": self.train_loss,
            "eval_loss": self.val_loss}, os.path.join(path, file_name))

