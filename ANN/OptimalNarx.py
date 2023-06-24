from torch.utils.data import DataLoader
from warnings import simplefilter
import matplotlib.pyplot as plt
from dataloader import *
from Trainer import *
import pandas as pd
import numpy as np
from ANN_Models import *
import torch

simplefilter(action='ignore', category=pd.errors.ParserWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)

data = DATA(10, 3) 
Xtrain, Ytrain = data.Xtrain, data.Ytrain

additional_loss=[]

# Load data and set them up correctly for dataloader
Dataset_train, Dataset_val = CustomDataset(Xtrain.to_numpy(), Ytrain.to_numpy()).split_data([0.8, 0.2])
Dataset_test = CustomDataset(np.array([1,1]), np.array([1,1]))
#Dataset_test = CustomDataset(data.testsub.u.to_numpy(), data.testsub.th.to_numpy())

dl_train = DataLoader(Dataset_train, batch_size=32, shuffle=True)
dl_val = DataLoader(Dataset_val, batch_size=32, shuffle=False)
dl_test = DataLoader(Dataset_test, batch_size=32, shuffle=True)

model = NARX(8)

train_module = Trainer(model, dl_train, dl_val, dl_test, "best_model.pt")
train_module.fit(100, 32)

