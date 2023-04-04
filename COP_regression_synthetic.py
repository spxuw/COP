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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
import xgboost as xgb
from scipy import stats
from torchmetrics import R2Score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet

parser = argparse.ArgumentParser(description='cnode')
parser.add_argument('--mode', default=None, help='mode')
parser.add_argument('--Nsub', default=None, help='N_sub')
parser.add_argument('--fold', default=None, help='fold')
parser.add_argument('--C', default=None, help='C')
parser.add_argument('--f', default=None, help='f')
parser.add_argument('--beta', default=None, help='beta')
parser.add_argument('--batch_size', default=None, help='batch_size')

args = parser.parse_args()
mode = args.mode
N_sub = int(args.Nsub)
Connectivity = (args.C)
fraction = (args.f)
beta = float(args.beta)
batch_size = int(args.batch_size)
fold = args.fold

################### load data ###########################
if mode == 'relative':
    X_train = np.loadtxt('../data/gLV/sigma_0.1/regression/gLV_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_X_train_relative_'+str(fraction)+'_'+str(fold)+'.csv', delimiter=',')
    X_test = np.loadtxt('../data/gLV/sigma_0.1/regression/gLV_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_X_test_relative_'+str(fraction)+'_'+str(fold)+'.csv', delimiter=',')
    y_train = np.loadtxt('../data/gLV/sigma_0.1/regression/gLV_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_y_train_relative_'+str(fraction)+'_'+str(fold)+'.csv', delimiter=',')
    y_test = np.loadtxt('../data/gLV/sigma_0.1/regression/gLV_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_y_test_relative_'+str(fraction)+'_'+str(fold)+'.csv', delimiter=',')
if mode == 'absolute':
    if str(fraction)!=0.5:
        X_train = np.loadtxt('../data/gLV/sigma_0.1/regression/gLV_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_X_train_absolute_'+str(fraction)+'_'+str(fold)+'.csv', delimiter=',')
        X_test = np.loadtxt('../data/gLV/sigma_0.1/regression/gLV_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_X_test_absolute_'+str(fraction)+'_'+str(fold)+'.csv', delimiter=',')
        y_train = np.loadtxt('../data/gLV/sigma_0.1/regression/gLV_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_y_train_absolute_'+str(fraction)+'_'+str(fold)+'.csv', delimiter=',')
        y_test = np.loadtxt('../data/gLV/sigma_0.1/regression/gLV_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_y_test_absolute_'+str(fraction)+'_'+str(fold)+'.csv', delimiter=',')
    if str(fraction)==0.5:
        X_train = np.loadtxt('../data/gLV/sigma_0.1/regression/gLV_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_X_train_absolute_'+str(1)+'_'+str(fold)+'.csv', delimiter=',')
        X_test = np.loadtxt('../data/gLV/sigma_0.1/regression/gLV_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_X_test_absolute_'+str(1)+'_'+str(fold)+'.csv', delimiter=',')
        y_train = np.loadtxt('../data/gLV/sigma_0.1/regression/gLV_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_y_train_absolute_'+str(1)+'_'+str(fold)+'.csv', delimiter=',')
        y_test = np.loadtxt('../data/gLV/sigma_0.1/regression/gLV_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_y_test_absolute_'+str(1)+'_'+str(fold)+'.csv', delimiter=',')

# For random forest
X_train_RF = X_train
X_test_RF = X_test
y_train_RF = y_train

# split training samples
number_of_cols = X_train.shape[1]
random_indices = np.random.choice(number_of_cols, size=int(0.2*number_of_cols), replace=False)
X_val = X_train[:,random_indices]
if str(fraction)==0.5:
    X_train =  X_train[:,np.setdiff1d(range(0,number_of_cols),random_indices)[0:50]]
if str(fraction)!=0.5:
    X_train =  X_train[:,np.setdiff1d(range(0,number_of_cols),random_indices)]

y_val = y_train[random_indices]
if str(fraction)==0.5:
    y_train = y_train[np.setdiff1d(range(0,number_of_cols),random_indices)[0:50]]
if str(fraction)!=0.5:
    y_train = y_train[np.setdiff1d(range(0,number_of_cols),random_indices)]

X_train = np.transpose(X_train)
X_test = np.transpose(X_test)
y_train = np.transpose(y_train)
y_test = np.transpose(y_test)
X_val = np.transpose(X_val)
y_val = np.transpose(y_val)


M,N = X_train.shape
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
X_val = X_val.astype(np.float64)


X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val)


threshold = torch.min(torch.min(y_train[y_train>0]),torch.min(y_val[y_val>0]))/2.0
y_train[y_train==0] = threshold
y_test[y_test==0] =  threshold
y_val[y_val==0] =  threshold

# revise response of LR, RF, XGBoost
y_train_RF[y_train_RF==0] =  threshold.numpy()

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
    fc_layers = [torch.nn.ReLU(inplace=True),torch.nn.Linear(dim, 1),torch.nn.Sigmoid()]
    model = torch.nn.Sequential(*feature_layers, *fc_layers)
    opt = optim.Adam(model.parameters(),lr=0.01)
    return model, opt

def prediction(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test.float())
    preds = logits
    vx = y_test
    vy = torch.log10(preds)
    vx = torch.log10(torch.reshape(vx,(len(y_test),1)).float())
    criterion = torch.nn.SmoothL1Loss(beta=beta)
    loss_val = (criterion(vx, vy))
    return preds,loss_val

def loss_batch(model, xb, yb, opt=None):
    vx = yb
    vy = (model(xb.float()))
    vy = torch.log10(vy)
    vx = torch.log10(torch.reshape(vx,(len(vx),1)).float())
    criterion = torch.nn.SmoothL1Loss(beta=beta)
    loss = (criterion(vx, vy))

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
            preds,loss_val = prediction(model, validate_x, validate_y)
            r2score = R2Score()
            preds  = torch.log10(preds)
            validate_yy = torch.log10(validate_y)
            Rsquare = r2score(preds, torch.reshape(validate_yy,(len(validate_yy),1)).float())
            if Rsquare>Loss_opt:
                Loss_opt = Rsquare
                best_model = copy.deepcopy(model)

            #print(f"Training... epoch {Rsquare}")
    model = copy.deepcopy(best_model)
    return Loss_opt

bs = [batch_size]

odenet, odeopt = get_model(rtol=1e-10, atol=1e-10)
R2_val = fit(10000, odenet, odeopt, X_train, y_train,X_val,y_val,bs)
pred_test,loss_test = prediction(odenet, X_test, y_test)

# check directory
MYDIR = '../results/gLV/sigma_0.1/regression_v2'
CHECK_FOLDER = os.path.isdir(MYDIR)
if not CHECK_FOLDER:
    os.makedirs(MYDIR)
else:
    print(MYDIR, "folder already exists.")

np.savetxt(str(MYDIR)+'/'+'Test_NODE_'+str(mode)+'_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_'+str(fraction)+'_'+str(fold)+'_'+str(beta)+'_'+str(batch_size)+'.csv',np.concatenate((np.log10(torch.reshape(y_test,(len(y_test),1))), pred_test), axis=1),delimiter=',')
np.savetxt(str(MYDIR)+'/'+'R2_NODE_'+str(mode)+'_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_'+str(fraction)+'_'+str(fold)+'_'+str(beta)+'_'+str(batch_size)+'.csv',np.reshape(R2_val.numpy(),(1,1)),delimiter=',')
print('Neural ODE done')

if beta == 0.001 and batch_size==16:
    # random forest regression
    # hyperparameters
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    max_features = ['auto', 'sqrt',None]
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

    RF = RandomForestRegressor(random_state=42)
    RF_random = RandomizedSearchCV(estimator = RF, param_distributions = random_grid_RF, scoring="r2", n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    RF_random.fit(np.transpose(X_train_RF), np.log10(y_train_RF))
    Pred_RF = RF_random.predict(np.transpose(X_test_RF))
    np.savetxt(str(MYDIR)+'/'+'Test_RF_'+str(mode)+'_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_'+str(fraction)+'_'+str(fold)+'.csv',np.concatenate((np.log10(torch.reshape(y_test,(len(y_test),1))), np.reshape(Pred_RF,(len(Pred_RF),1))), axis=1),delimiter=',')
    print('Random forest done')

    # LR LinearRegression
    LR =  LinearRegression()
    LR.fit(np.transpose(X_train_RF), np.log10(y_train_RF))
    Pred_LR = LR.predict(np.transpose(X_test_RF))
    np.savetxt(str(MYDIR)+'/'+'Test_LR_'+str(mode)+'_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_'+str(fraction)+'_'+str(fold)+'.csv',np.concatenate((np.log10(torch.reshape(y_test,(len(y_test),1))), np.reshape(Pred_LR,(len(Pred_LR),1))), axis=1),delimiter=',')
    print('LR done')

    # Xgboost
    GBR =  xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    random_grid_XG = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'min_child_weight': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
        'gamma': [0, 0.01, 0.1, 1, 2, 5, 10, 20, 40],
        'learning_rate': [0.0005, 0.01, 0.1, 0.2, 0.3],
        'subsample': [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1]}

    GBR_random = RandomizedSearchCV(estimator = GBR, param_distributions = random_grid_XG, scoring="r2", n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    GBR_random.fit(np.transpose(X_train_RF), np.log10(y_train_RF))
    Pred_GBR = GBR_random.predict(np.transpose(X_test_RF))
    np.savetxt(str(MYDIR)+'/'+'Test_XG_'+str(mode)+'_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_'+str(fraction)+'_'+str(fold)+'.csv',np.concatenate((np.log10(torch.reshape(y_test,(len(y_test),1))), np.reshape(Pred_GBR,(len(Pred_GBR),1))), axis=1),delimiter=',')
    print('Xgboost done')

    # enet
    enet = ElasticNet()
    random_grid_enet = {
        "max_iter": [1, 5, 10,100,1000],
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        "l1_ratio": np.arange(0.0, 1.0, 0.1)}
    enet_random = RandomizedSearchCV(estimator = enet, param_distributions = random_grid_enet, scoring="r2", n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    enet_random.fit(np.transpose(X_train_RF), np.log10(y_train_RF))
    Pred_enet = enet_random.predict(np.transpose(X_test_RF))
    np.savetxt(str(MYDIR)+'/'+'Test_ENET_'+str(mode)+'_sub_'+str(N_sub)+'_C_'+str(Connectivity)+'_'+str(fraction)+'_'+str(fold)+'.csv',np.concatenate((np.log10(torch.reshape(y_test,(len(y_test),1))), np.reshape(Pred_enet,(len(Pred_enet),1))), axis=1),delimiter=',')
    print('Enet done')