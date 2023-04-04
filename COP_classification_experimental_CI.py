# author: Xu-Wen Wang <spxuw@channing.harvard.edu>
# last update: 2023-04-04

import torch
import numpy as np
import sys, copy, math, time, pdb
import os.path
import random
import pdb
import csv
import argparse
import itertools
import torch.optim as optim
from torchdiffeq import odeint
from torch.utils.data import TensorDataset, DataLoader
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import numpy_indexed as npi
import pandas as pd
from scipy import stats
import numpy.matlib
from torchmetrics import AUROC
from sklearn.model_selection import RandomizedSearchCV


parser = argparse.ArgumentParser(description='cnode')
parser.add_argument('--mode', default=None, help='mode')
parser.add_argument('--Triple', default=None, help='Triple')
parser.add_argument('--dataset', default=None, help='dataset')
parser.add_argument('--batch_size', default=None, help='batch_size')


args = parser.parse_args()
mode = args.mode
dataset = args.dataset
Triple = args.Triple
batch_size = int(args.batch_size)

################### load data ###########################
if mode == 'relative':
    X_train = np.loadtxt('../data/'+str(dataset)+'/X_train_relative_NI_triple_'+str(Triple)+'.csv', delimiter=',')
    X_test = np.loadtxt('../data/'+str(dataset)+'/X_test_relative_NI_triple_'+str(Triple)+'.csv', delimiter=',')
    y_train = np.loadtxt('../data/'+str(dataset)+'/y_train_relative_NI_triple_'+str(Triple)+'.csv', delimiter=',')
if mode == 'absolute':
    X_train = np.loadtxt('../data/'+str(dataset)+'/X_train_absolute_NI_triple_'+str(Triple)+'.csv', delimiter=',')
    X_test = np.loadtxt('../data/'+str(dataset)+'/X_test_absolute_NI_triple_'+str(Triple)+'.csv', delimiter=',')
    y_train = np.loadtxt('../data/'+str(dataset)+'/y_train_absolute_NI_triple_'+str(Triple)+'.csv', delimiter=',')

# binary 
y_train[y_train>0] = 1

X_train_RF = X_train
X_test_RF = X_test
y_train_RF = y_train

# split training samples
number_of_cols = X_train.shape[1]
random_indices = np.random.choice(number_of_cols, size=int(0.2*number_of_cols), replace=False)
X_val = X_train[:,random_indices]
X_train =  X_train[:,np.setdiff1d(range(0,number_of_cols),random_indices)]


y_val = y_train[random_indices]
y_train = y_train[np.setdiff1d(range(0,number_of_cols),random_indices)]


X_train = np.transpose(X_train)
X_test = np.transpose(X_test)
y_train = np.transpose(y_train)
X_val = np.transpose(X_val)
y_val = np.transpose(y_val)

# without spliting
X_train_all = np.transpose(X_train_RF)
y_train_all = np.transpose(y_train_RF)


M,N = X_train.shape
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
X_val = X_val.astype(np.float64)


X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val)

X_train_all = torch.from_numpy(X_train_all)
y_train_all = torch.from_numpy(y_train_all)


def norm(dim):
    """
    Group normalization to improve model accuracy and training speed.
    """
    return torch.nn.GroupNorm(min(dim, dim), dim)

def get_batch(X_train,y_train,mb_size):
    #pdb.set_trace()
    s = torch.from_numpy(np.random.choice(np.arange(X_train.size(dim=0), dtype=np.int64), mb_size, replace=False))
    batch_p = X_train[s,:]
    batch_q = y_train[s]
    return batch_p, batch_q

class ODEFunc(torch.nn.Module):
    def __init__(self,dim):
        super(ODEFunc, self).__init__()
        self.norm1 = norm(N)
        self.norm2 = norm(N)
        self.fcc1 = torch.nn.Linear(N, N)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.fcc2 = torch.nn.Linear(N, N)
        self.relu2 = torch.nn.ReLU(inplace=True)
    def forward(self, t, y):
        out = self.fcc1(y)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.fcc2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        return out

class ODENet(torch.nn.Module):
    def __init__(self, ODEFunc, rtol=1e-10, atol=1e-10):
        super(ODENet, self).__init__()
        self.ODEFunc = ODEFunc
        self.integration_time = torch.tensor([0, 10]).float()
        self.rtol = rtol
        self.atol = atol

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.ODEFunc, x, self.integration_time,rtol=self.rtol, atol=self.atol)
        return out[1]

def get_model(dim=N, **kwargs):
    feature_layers = [ODENet(ODEFunc(N), **kwargs)]
    fc_layers = [torch.nn.ReLU(inplace=True),torch.nn.Linear(dim, 64),torch.nn.ReLU(),torch.nn.Linear(64, 2),torch.nn.Sigmoid()]
    model = torch.nn.Sequential(*feature_layers, *fc_layers)
    opt = optim.Adam(model.parameters(),lr=0.01)
    return model, opt

def prediction(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test.float())
    preds = logits
    vy = preds
    vx = y_test
    auroc = AUROC(pos_label=1)
    AUC = auroc(torch.softmax(vy,dim=1)[:,1],torch.as_tensor(vx.int().tolist()))
    return AUC,torch.softmax(vy,dim=1)[:,1]

def loss_batch(model, xb, yb, opt=None):
    vx = yb
    vy = (model(xb.float()))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(vy, torch.as_tensor(vx.int().tolist()))

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

def fit(epochs, model, opt, train_x, train_y, validate_x, validate_y, bs):
    Loss_opt = -10
    batch_range = bs
    for bs in batch_range:
        for epoch in range(epochs):
            model.train()   # Set model to training mode
            xb,yb = get_batch(train_x,train_y,bs)
            loss_epoch, batch_size = loss_batch(model, xb, yb, opt)
            AUC_val,pred_val = prediction(model, validate_x, validate_y)
            if AUC_val>Loss_opt:
                Loss_opt = AUC_val
                best_model = copy.deepcopy(model)

            #print(f"Training... epoch {AUC_val}")
    model = copy.deepcopy(best_model)
    return Loss_opt


bs = [batch_size]

odenet, odeopt = get_model(rtol=1e-10, atol=1e-10)
AUC_val = fit(10000, odenet, odeopt, X_train, y_train,X_val,y_val,bs)
AUC_test,pred_test = prediction(odenet, X_test, X_test[:,0])
AUC_train,pred_train = prediction(odenet, X_train_all, y_train_all)

# check directory
MYDIR = '../results/'+str(dataset)+'/classification_cocktail_personlized_v2'
CHECK_FOLDER = os.path.isdir(MYDIR)
if not CHECK_FOLDER:
    os.makedirs(MYDIR)
else:
    print(MYDIR, "folder already exists.")

np.savetxt(str(MYDIR)+'/'+'AUC_NODE_'+str(mode)+'_'+str(Triple)+'_'+str(batch_size)+'.csv',np.reshape(AUC_val.numpy(),(1,1)),delimiter=',')
# cocktail recommendation via adding absent species
# step 1: calculate net impact
sample_id = np.loadtxt('../data/'+str(dataset)+'/sample_id_triple_'+str(Triple)+'.csv', delimiter=',',dtype=int)
species_id = np.loadtxt('../data/'+str(dataset)+'/species_id_triple_'+str(Triple)+'.csv', delimiter=',',dtype=int)
species_overlap = pd.read_csv('../data/'+str(dataset)+'/species_name_overlap_triple_'+str(Triple)+'.csv', sep=',',header=None)
species_name = pd.read_csv('../data/'+str(dataset)+'/species_name_new_triple_'+str(Triple)+'.csv', sep=',',header=None)


pred_train = pred_train[y_train_RF>0]
y_test_before = pred_train[sample_id-1] # use predicted as baseline, rather than true abundance before perturbation
#y_test_before = y_test_before[np.where(y_train_RF>0)]
pred_test = np.reshape(pred_test,y_test_before.shape)
y_test_before = y_test_before
y_test_before = np.reshape(y_test_before,y_test_before.shape)
relative_yield_NODE = (pred_test-y_test_before)/(y_test_before + pred_test)
#groups_NODE, means_NODE = npi.group_by(species_id).median(relative_yield_NODE)


## step 2: combine different prediction methods and identify global promotors and inhibitors
data_RY = {'Species':species_id,'Sample': sample_id,'NODE':np.reshape(relative_yield_NODE,(relative_yield_NODE.shape[0],))}
df_RY = pd.DataFrame(data_RY)
df_RY.to_csv(str(MYDIR)+'/'+'RY_NODE_'+str(mode)+'_'+str(Triple)+'_'+str(batch_size)+'.csv', index=False)

   
# other methods
if batch_size==16:
    LR =  LogisticRegression()
    LR.fit(np.transpose(X_train_RF), (y_train_RF))

    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid_RF = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    RF = RandomForestClassifier(random_state=42)
    RF_random = RandomizedSearchCV(estimator = RF, param_distributions = random_grid_RF, scoring="roc_auc", n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    RF_random.fit(np.transpose(X_train_RF), (y_train_RF))

    
    GBR =  xgb.XGBClassifier(random_state=42)
    random_grid_XG = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'min_child_weight': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
        'gamma': [0, 0.01, 0.1, 1, 2, 5, 10, 20, 40],
        'learning_rate': [0.0005, 0.01, 0.1, 0.2, 0.3],
        'subsample': [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1]}

    GBR_random = RandomizedSearchCV(estimator = GBR, param_distributions = random_grid_XG, scoring="roc_auc", n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    GBR_random.fit(np.transpose(X_train_RF), (y_train_RF))

    pred_test_LR = LR.predict_proba(np.transpose(X_test_RF))[:,1]
    pred_test_RF = RF_random.predict_proba(np.transpose(X_test_RF))[:,1]
    pred_test_XG = GBR_random.predict_proba(np.transpose(X_test_RF))[:,1]
    pred_train_LR = LR.predict_proba(np.transpose(X_train_RF))[:,1][y_train_RF>0]
    pred_train_RF = RF_random.predict_proba(np.transpose(X_train_RF))[:,1][y_train_RF>0]
    pred_train_XG = GBR_random.predict_proba(np.transpose(X_train_RF))[:,1][y_train_RF>0]

    
    y_test_before_LR = pred_train_LR[sample_id-1]
    y_test_before_RF = pred_train_RF[sample_id-1]
    y_test_before_XG = pred_train_XG[sample_id-1]

    relative_yield_LR = (pred_test_LR-y_test_before_LR)/(pred_test_LR + y_test_before_LR)
    relative_yield_RF = (pred_test_RF-y_test_before_RF)/(pred_test_RF + y_test_before_RF)
    relative_yield_XG = (pred_test_XG-y_test_before_XG)/(pred_test_XG + y_test_before_XG)

    data_RY_LR = {'Species':species_id,'Sample': sample_id,'LR':np.reshape(relative_yield_LR,(relative_yield_LR.shape[0],))}
    data_RY_LR = pd.DataFrame(data_RY_LR)
    data_RY_LR.to_csv(str(MYDIR)+'/'+'RY_LR_'+str(mode)+'_'+str(Triple)+'.csv', index=False)

    data_RY_RF = {'Species':species_id,'Sample': sample_id,'RF':np.reshape(relative_yield_RF,(relative_yield_RF.shape[0],))}
    data_RY_RF = pd.DataFrame(data_RY_RF)
    data_RY_RF.to_csv(str(MYDIR)+'/'+'RY_RF_'+str(mode)+'_'+str(Triple)+'.csv', index=False)

    data_RY_XG = {'Species':species_id,'Sample': sample_id,'XG':np.reshape(relative_yield_XG,(relative_yield_XG.shape[0],))}
    data_RY_XG = pd.DataFrame(data_RY_XG)
    data_RY_XG.to_csv(str(MYDIR)+'/'+'RY_XG_'+str(mode)+'_'+str(Triple)+'.csv', index=False)

    data_RY_ENET = {'Species':species_id,'Sample': sample_id,'XG':np.reshape(relative_yield_ENET,(relative_yield_ENET.shape[0],))}
    data_RY_ENET = pd.DataFrame(data_RY_XG)
    data_RY_ENET.to_csv(str(MYDIR)+'/'+'RY_ENET_'+str(mode)+'_'+str(Triple)+'.csv', index=False)
