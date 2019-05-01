import numpy as np
import numpy.matlib
import sys
import pickle
from collections import Counter
# from scoop import futures
from math import log,sqrt

# idx = int(sys.argv[1])
N = int(sys.argv[1])
m = int(sys.argv[2])
number_layers = int(sys.argv[3])
sigmaw = float(sys.argv[4])
sigmab = float(sys.argv[5])
nonlin="relu"
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#
# output_dim=1
# paral_nets=1000

signature_string = str(m)+"_"+str(sigmaw)+"_"+str(sigmab)+"_"+str(number_layers)+"_"+str(N*size)

inputs = np.load(open( "10_fc_mnist_0.0_train_images.np", "rb" ))
input_dim = inputs.shape[1]
hidden_dim=input_dim

def sample_net():
        number_outputs = []
        Hs = []
        betas = []
        X = inputs.T
        preac = np.matmul((sigmaw/np.sqrt(input_dim))*np.random.randn(hidden_dim,input_dim),X) \
                        + np.matlib.repmat(sigmab*np.random.randn(hidden_dim,1),1,m)
        outs = np.heaviside(preac,0)
        # H, N_O = calc_entropy(outs.T)
        # beta = 2**H/N_O
        # Hs.append(H)
        # betas.append(beta)
        # number_outputs.append(N_O)
        if nonlin == "relu":
                X = np.maximum(0,preac)
        elif nonlin == "step":
                X = np.heaviside(preac,0)
        elif nonlin == "tanh":
                X = np.tanh(preac)
        for l in range(1,number_layers):
                preac = np.matmul((sigmaw/np.sqrt(hidden_dim))*np.random.randn(hidden_dim,hidden_dim),X) \
                                + np.matlib.repmat(sigmab*np.random.randn(hidden_dim,1),1,m)
                outs = np.heaviside(preac,0)
                # H, N_O = calc_entropy(outs.T)
                # beta = 2**H/N_O
                # Hs.append(H)
                # betas.append(beta)
                # number_outputs.append(N_O)
                if nonlin == "relu":
                        X = np.maximum(0,preac)
                elif nonlin == "step":
                        X = np.heaviside(preac,0)
                elif nonlin == "tanh":
                        X = np.tanh(preac)
        preac = np.matmul((sigmaw/np.sqrt(hidden_dim))*np.random.randn(1,hidden_dim),X) \
                        + np.matlib.repmat(sigmab*np.random.randn(1,1),1,m)
        outs = np.heaviside(preac,0)
        outs_str = "".join([str(int(y[0])) for y in outs.T.tolist()])
        return outs_str

outs_str = []
for i in range(N):
    outs_str.append(sample_net())

cnts = Counter(outs_str)
pickle.dump(cnts,open("logPUcnts/NN_cnts_"+str(rank)+"_"+signature_string+".p","wb"))

#outs_str = comm.gather(outs_str,root=0)
#
#if rank == 0:
#    outs_str = sum(outs_str,[])
#
#    cnts = Counter(outs_str)
#    pickle.dump(cnts,open("logPUcnts/NN_cnts_"+signature_string+".p","wb"))

