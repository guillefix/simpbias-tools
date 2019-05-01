import numpy as np
import numpy.matlib
import sys
import pickle
from collections import Counter
# from scoop import futures
from math import log,sqrt

hidden_layers = int(sys.argv[1])
nonlin = sys.argv[2]
#sigmaw=1/sqrt(20)
sigmaw=5 #note that I just divided by input_dim, to agree with notation in deep info prop papers..
sigmab=0.05

input_dim=7
hidden_dim=40
inputs = [[float(x) for x in "{0:b}".format(i).zfill(input_dim)] for i in range(0,2**input_dim)]

N=2**input_dim
logN = log(N)
def calc_entropy(outs):
    cnts=Counter(map(lambda x: "".join([str(int(y)) for y in x]),outs.tolist()))
    ent = 0
    for key,val in cnts.most_common():
        p = 1.0*val/N
        logp = (log(val)-log(N))/log(2)
        ent += -p*logp
    return ent,len(cnts)

def sample_net():
	number_outputs = []
	Hs = []
	betas = []
	X = np.array(inputs).T
	preac = np.matmul((sigmaw/input_dim)*np.random.randn(hidden_dim,input_dim),X) \
			+ np.matlib.repmat(sigmab*np.random.randn(hidden_dim,1),1,2**input_dim)
	outs = np.heaviside(preac,0)
	H, N_O = calc_entropy(outs.T)
	beta = 2**H/N_O
	Hs.append(H)
	betas.append(beta)
	number_outputs.append(N_O)
	if nonlin == "relu":
		X = np.maximum(0,preac)
	elif nonlin == "step":
		X = np.heaviside(preac,0)
	elif nonlin == "tanh":
		X = np.tanh(preac)
	for l in range(1,hidden_layers+1):
		preac = np.matmul((sigmaw/hidden_dim)*np.random.randn(hidden_dim,hidden_dim),X) \
				+ np.matlib.repmat(sigmab*np.random.randn(hidden_dim,1),1,2**input_dim)
		outs = np.heaviside(preac,0)
		H, N_O = calc_entropy(outs.T)
		beta = 2**H/N_O
		Hs.append(H)
		betas.append(beta)
		number_outputs.append(N_O)
		if nonlin == "relu":
			X = np.maximum(0,preac)
		elif nonlin == "step":
			X = np.heaviside(preac,0)
		elif nonlin == "tanh":
			X = np.tanh(preac)
	preac = np.matmul((sigmaw/hidden_dim)*np.random.randn(1,hidden_dim),X) \
			+ np.matlib.repmat(sigmab*np.random.randn(1,1),1,2**input_dim)
	outs = np.heaviside(preac,0)
	outs_str = "".join([str(int(y[0])) for y in outs.T.tolist()])
	return Hs, betas, number_outputs, outs_str

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)

print("Doing task %d of %d" % (rank, size))

Hs, betas, number_outputs, outs_str = sample_net()

Hs = comm.gather(Hs,root=0)
betas = comm.gather(betas,root=0)
number_outputs = comm.gather(number_outputs,root=0)
outs_str = comm.gather(outs_str,root=0)
#print(data)
if rank == 0:
	signature_str = str(nonlin)+"_"+str(sigmaw)+"_"+str(sigmab)+"_"+str(size)
	Hs = np.array(Hs).T.tolist()
	betas = np.array(betas).T.tolist()
	number_outputs = np.array(number_outputs).T.tolist()
	pickle.dump(Hs,open("Hs_"+signature_str+".p","wb"))
	pickle.dump(betas,open("betas_"+signature_str+".p","wb"))
	pickle.dump(number_outputs,open("number_outputs_"+signature_str+".p","wb"))
	pickle.dump(outs_str,open("outs_str_"+signature_str+".p","wb"))
	
	
 
