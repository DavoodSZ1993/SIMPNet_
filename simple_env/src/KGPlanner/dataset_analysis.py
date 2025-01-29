#!/usr/bin/env python3

import pandas as pd 
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

eps=1e-5
all_features = torch.load('all_features.pt')
all_labels = torch.load('all_labels.pt')
all_obstacles = torch.load('all_obs_embed.pt')
print(f'Shape of all features: {all_features.shape}')
print(f'shape pf all labels: {all_labels.shape}')
all_data = torch.cat((all_features, all_labels), dim=1)
print(f'Shape of all data: {all_data.shape}')


all_data_mean = torch.mean(all_data, dim=0)
all_data_std = torch.std(all_data, dim=0)
all_data = (all_data - all_data_mean) / (all_data_std  + eps)

df = pd.DataFrame(all_data.numpy())

'''
for column in df.columns:
    plt.figure()
    sns.boxplot(x=df[column])
    plt.title(column)
    plt.show()'''


correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Correlation coefficient'})
plt.title('Correlation matrix between features and labels.')

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix.iloc[16:, :16], annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Correlation coefficient'})
plt.title('Correlation matrix between features and labels.')
plt.show()
