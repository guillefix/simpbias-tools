import pickle
import numpy as np

weights = pickle.load(open("weights_1000000_7_20_1_allinputs_relu.p","rb"))

with open("weights_1000000_7_20_1_relu.txt","w") as f:
	for fun, ws in weights.iteritems():
		w=ws[0]
		inital_weights1 = w[0].flatten(order='F')
		inital_biases1 = w[1].flatten(order='F')
		inital_weights2 = w[2].flatten(order='F')
		inital_biases2 = w[3].flatten(order='F')
		#inital_weights3 = w[4].flatten(order='F')
		#inital_biases3 = w[5].flatten(order='F')
		#w_concat = np.concatenate((inital_weights1, inital_biases1, inital_weights2, inital_biases2, inital_weights3, inital_biases3))
		w_concat = np.concatenate((inital_weights1, inital_biases1, inital_weights2, inital_biases2))
		f.write(fun+"\t"+" ".join(["%.8f" % n for n in w_concat])+"\n")
	
