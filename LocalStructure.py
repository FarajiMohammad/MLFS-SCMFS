from sklearn.neighbors import kneighbors_graph
import numpy as np
import torch
def LCS(X,Si,nk):

    S = kneighbors_graph(X, nk, mode='distance', include_self=False, metric='euclidean').todense()
    S[S==0]=np.inf
    S = np.array(np.exp(-S/(Si)))
    S = torch.from_numpy(S).to(torch.float32)
    #Dii
    Ds = torch.diag(torch.sum(S, dim=1))
    return S,Ds