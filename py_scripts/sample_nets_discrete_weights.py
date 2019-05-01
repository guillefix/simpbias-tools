import tensorflow as tf
import numpy as np
# from sklearn.metrics import confusion_matrix
# import time
# from datetime import timedelta
import math
from collections import Counter
np.set_printoptions(threshold=np.nan)
from KC_LZ import calc_KC
import pickle
import sys

#(idx, N) = [int(ar) for ar in sys.argv[1:]]

idx = int(sys.argv[1])

input_dim=7
hidden_layer_dim=20
hidden_layer2_dim=20
hidden_layer_dim2 = hidden_layer2_dim
output_dim=1

## VARIABLE declarations
#for many paralell networks

from math import sqrt
W1=[]
b1=[]
W2=[]
b2=[]
W3=[]
b3=[]
variables = []

paral_nets = 5000

a=sqrt(3)*sqrt(2)
b=sqrt(3)
discrete_step=1

for i in range(paral_nets):
    scope_name = "net"+str(i)
    with tf.variable_scope(scope_name):
        W1.append(tf.Variable(np.random.choice(np.arange(-10,10,discrete_step,dtype=np.float16),(input_dim,hidden_layer_dim))))
        b1.append(tf.Variable(np.random.choice(np.arange(-10,10,discrete_step,dtype=np.float16),(hidden_layer_dim))))

        W2.append(tf.Variable(np.random.choice(np.arange(-10,10,discrete_step,dtype=np.float16),(hidden_layer_dim,hidden_layer_dim2))))
        b2.append(tf.Variable(np.random.choice(np.arange(-10,10,discrete_step,dtype=np.float16),(hidden_layer_dim2))))

#         W2.append(tf.Variable(tf.random_uniform([hidden_layer_dim,output_dim],-a/sqrt(hidden_layer_dim),a/sqrt(hidden_layer_dim))))
#         b2.append(tf.Variable(tf.random_uniform([output_dim],-a/sqrt(hidden_layer_dim),a/sqrt(hidden_layer_dim))))

        W3.append(tf.Variable(np.random.choice(np.arange(-10,10,discrete_step,dtype=np.float16),(hidden_layer_dim2,output_dim))))
        b3.append(tf.Variable(np.random.choice(np.arange(-10,10,discrete_step,dtype=np.float16),(output_dim))))

#         W1.append(tf.Variable(tf.random_normal([input_dim,hidden_layer_dim],stddev=1/sqrt(input_dim))))
#         b1.append(tf.Variable(tf.random_normal([hidden_layer_dim],stddev=1/sqrt(input_dim))))

#         W2.append(tf.Variable(tf.random_normal([hidden_layer_dim,hidden_layer2_dim],stddev=1/sqrt(hidden_layer_dim))))
#         b2.append(tf.Variable(tf.random_normal([hidden_layer2_dim],stddev=1/sqrt(hidden_layer_dim))))

#         W3.append(tf.Variable(tf.random_normal([hidden_layer2_dim,output_dim],stddev=1/sqrt(hidden_layer2_dim))))
#         b3.append(tf.Variable(tf.random_normal([output_dim],stddev=1/sqrt(hidden_layer2_dim))))

        variables.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name))

x = tf.placeholder(tf.float16, shape=[None, input_dim], name='x')

## NETWORK construction

outputs = []

for i in range(paral_nets):
    h = tf.matmul(x, W1[i]) + b1[i]
#     h = tf.matmul(x, W1[i])
#     h = tf.sign(h)
    h = tf.nn.relu(h)
    h2 = tf.matmul(h, W2[i]) + b2[i]
    # h2 = tf.sign(h2)
    h2 = tf.nn.relu(h2)
    logits = tf.matmul(h2, W3[i]) + b3[i]
#     logits = tf.matmul(h, W2[i]) + b2[i]
#     logits = tf.matmul(h, W2[i])
    o = tf.sign(logits)
#     outputs.append((o+1)/2)
    outputs.append(tf.reduce_join(tf.reduce_join(tf.as_string(tf.cast((o+1)//2,tf.int8)), 0),0))

session = tf.Session()

train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
param_shape = []
val_placeholders = []
ops=[]
param_size=[]
for i,var in enumerate(train_vars):
        param_shape.append(tuple(var.get_shape().as_list()))
        param_size.append(np.prod(var.get_shape().as_list()))
        val_placeholders.append(tf.placeholder(tf.float16, shape = param_shape[i], name="val_"+str(i)))
        ops.append(var.assign_add(val_placeholders[i]))

def get_param_vec():
        params = [p.flatten() for p in session.run(train_vars)]
        return np.concatenate(params)

def update_params(params_change):
	j = 0
	change_feed_dict = {}
	for i,var in enumerate(train_vars):
		#print(i)
		val_change = params_change[j:j+param_size[i]]
		j += param_size[i]
		val_change = val_change.reshape(param_shape[i])
		change_feed_dict["val_"+str(i)+":0"]=val_change
	session.run(ops,feed_dict=change_feed_dict)

inputs = [[float(xx) for xx in "{0:07b}".format(i)] for i in range(2**7)]

# N=10
#cnt = Counter()
#weights = {}
#for i in range(N):
    # if i%(N/100) == 0:
session.run(tf.global_variables_initializer())
fs = session.run(outputs, feed_dict={x:inputs})
#phenos = [[] for i in range(paral_nets)]
phenos = [fs[i] for i in range(paral_nets)]
#varss = session.run(variables,feed_dict={x:inputs})
    #for i,f in enumerate(fs):
    #    if f in weights:
    #        weights[f].append(varss[i])
    #    else:
    #        weights[f]=[varss[i]]

robs=[0.0 for x in phenos]
cnt = Counter(fs)
#phenos = [x+[fs[i]] for i,x in enumerate(phenos)]
param_num=input_dim*hidden_layer_dim + hidden_layer_dim + hidden_layer_dim*hidden_layer2_dim + hidden_layer2_dim + hidden_layer2_dim*output_dim + output_dim
change_vec_ind = np.zeros(param_num)
for i in range(param_num):
	print(str(i+1)+"/"+str(param_num))
	change_vec_ind[i] = discrete_step
	change_vec = np.concatenate([change_vec_ind for j in range(paral_nets)])
	update_params(change_vec)
	#session.run(tf.global_variables_initializer())
	fs = session.run(outputs, feed_dict={x:inputs})
	#phenos = [x+[fs[j]] for j,x in enumerate(phenos)]
	robs = [xx+(1.0 if fs[j]==phenos[j] else 0.0) for j,xx in enumerate(robs)]
	change_vec_ind[i] = -discrete_step
	change_vec = np.concatenate([change_vec_ind for j in range(paral_nets)])
	update_params(change_vec)
	change_vec_ind[i] = 0

#robs=[]
freqs=[]
for i,p in enumerate(phenos):
	robs[i] = robs[i]/param_num
	freqs.append(cnt[p])


pickle.dump(cnt, open( str(idx)+"_cnt_"+str(paral_nets)+"_"+str(input_dim)+"_"+str(hidden_layer_dim)+"_"+str(hidden_layer2_dim)+"_"+str(output_dim)+"_"+str(discrete_step)+"_relu.p", "wb" ), -1)
pickle.dump(phenos, open( str(idx)+"_phenos_"+str(paral_nets)+"_"+str(input_dim)+"_"+str(hidden_layer_dim)+"_"+str(hidden_layer2_dim)+"_"+str(output_dim)+"_"+str(discrete_step)+"_relu.p", "wb" ), -1)
pickle.dump(robs, open( str(idx)+"_robs_"+str(paral_nets)+"_"+str(input_dim)+"_"+str(hidden_layer_dim)+"_"+str(hidden_layer2_dim)+"_"+str(output_dim)+"_"+str(discrete_step)+"_relu.p", "wb" ), -1)
pickle.dump(freqs, open( str(idx)+"_freqs"+str(paral_nets)+"_"+str(input_dim)+"_"+str(hidden_layer_dim)+"_"+str(hidden_layer2_dim)+"_"+str(output_dim)+"_"+str(discrete_step)+"_relu.p", "wb" ), -1)
#pickle.dump(weights, open( str(idx)+"_weights_"+str(N*paral_nets)+"_"+str(input_dim)+"_"+str(hidden_layer_dim)+"_"+str(hidden_layer2_dim)+"_"+"sallinputs_relu.p", "wb" ), -1)

#with open(str(idx)+"_comp_freq_7_20_20_1_relu", "w") as f:
#    for fun,val in cnt.most_common():
#        f.write(str(calc_KC(str(fun)))+"\t"+str(val)+"\n")

