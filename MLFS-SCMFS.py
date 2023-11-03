import torch
from tqdm import tqdm
import numpy as np
from skmultilearn.dataset import load_dataset
from skmultilearn.dataset import available_data_sets
from scipy.io import loadmat
from sklearn.cluster import KMeans
from skmultilearn.adapt import MLkNN
from sklearn.neighbors import kneighbors_graph
import sklearn
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import pandas as pd
# Metrics
from sklearn.metrics import coverage_error
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import hamming_loss
import scipy.sparse as sp
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

## GPU or CPU
GPU = False
if GPU:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("num GPUs", torch.cuda.device_count())
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
    print("CPU")



datas =['Arts','Business','Computers','Entertainment','Recreation','Society']
dat = loadmat('Datasets/' + datas[0] + ".mat")
train = dat['train']
test = dat['test']

X_test = torch.from_numpy(train[0][0].T).to(torch.float32)
Y_test = torch.from_numpy(train[0][1].T).to(torch.float32)
Y_test[Y_test == -1] = 0

X = torch.from_numpy(test[0][0].T).to(torch.float32).type(dtype)
Xc = torch.from_numpy(test[0][0].T).to(torch.float32)
XTX = X.T @ X

Yc = torch.from_numpy(test[0][1].T).to(torch.float32)
Yc[Yc == -1] = 0

Y = Yc.type(dtype)
# Feature
n, d = X.shape
# label
n, c = Y.shape

# Number of selected Latent
# coefficient Matrix
# Number of selected Latent
k = 23
# coefficient Matrix
W = torch.rand(d, k).type(dtype)
V = torch.rand(n, k).type(dtype)
Q = torch.rand(k, d).type(dtype)
B = torch.rand(k, c).type(dtype)
D = torch.diag(1 / (2 * torch.linalg.norm(W, axis=1)))
# Number of Iteration & Parameters
iteration = 60

alpha = 0.1
beta = 0.1
gamma = 0.1
epsilon = torch.tensor(torch.finfo(torch.float16).eps)

# Number of Iteration & Parameters
for j in range(iteration):
    Wu = (X.T @ V)
    Wd = (X.T @ X @ W) + gamma * (D @ W)
    W = (W * (Wu) / torch.maximum(Wd, epsilon))
    Vu = (X @ W + (alpha * (X @ Q.T)) + (beta * (Y @ B.T)))
    Vd = (V + (alpha * (V @ Q @ Q.T)) + (beta * (V @ B @ B.T)))
    V = (V * ((Vu) / torch.maximum((Vd), epsilon)))
    Qu = (V.T @ X)
    Qd = (V.T @ V @ Q)
    Q = Q * (Qu) / torch.maximum(Qd, epsilon)
    Bu = (V.T @ Y)
    Bd = (V.T @ V @ B)
    B = B * ((Bu) / torch.maximum((Bd), epsilon))
    D = torch.diag(1 / (2 * torch.linalg.norm(W, axis=1) + epsilon))
w = torch.linalg.norm(W, axis=1)
sQ = torch.argsort(w)

for j in tqdm(range(20)):
    j += 1
    nosf = int(j * d / 100)
    sX = Xc[:, sQ[d - nosf:].long()]
    classifier = MLkNN(k=10)
    classifier.fit(sX.numpy(), Yc.numpy())
    # KNN
    predictions = classifier.predict(X_test[:, sQ[d - nosf:]]).toarray()
    scores = classifier.predict_proba(X_test[:, sQ[d - nosf:]]).toarray()

    Mic = f1_score(Y_test, predictions, average='micro')
    Mac = f1_score(Y_test, predictions, average='macro')
    HML = hamming_loss(Y_test, predictions)
    RNL = label_ranking_loss(Y_test, scores)
    AVP = average_precision_score(Y_test, scores)
    COV = coverage_error(Y_test, scores)
print('Micro-F1:',Mic,'\n','Macro-F1:',Mac,'\n','Hamming Loss:',HML,'\n','Ranking Loss:',RNL,'\n','Average Precision:',AVP,'\n','Coverage Error:',COV)

