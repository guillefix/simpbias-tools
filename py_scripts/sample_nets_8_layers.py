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

(idx, N) = [int(ar) for ar in sys.argv[1:]]

input_dim=7
hidden_layer_dim=40

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
W4=[]
b4=[]
W5=[]
b5=[]
W6=[]
b6=[]
W7=[]
b7=[]
W8=[]
b8=[]
W9=[]
b9=[]
variables = []

paral_nets = 500

a=sqrt(3)*sqrt(2)
b=sqrt(3)

for i in range(paral_nets):
    scope_name = "net"+str(i)
    with tf.variable_scope(scope_name):
        W1.append(tf.Variable(tf.random_uniform([input_dim,hidden_layer_dim],-a/sqrt(input_dim),a/sqrt(input_dim))))
        b1.append(tf.Variable(tf.random_uniform([hidden_layer_dim],-a/sqrt(input_dim),a/sqrt(input_dim))))

        W2.append(tf.Variable(tf.random_uniform([hidden_layer_dim,hidden_layer_dim],-a/sqrt(hidden_layer_dim),a/sqrt(hidden_layer_dim))))
        b2.append(tf.Variable(tf.random_uniform([hidden_layer_dim],-a/sqrt(hidden_layer_dim),a/sqrt(hidden_layer_dim))))

#         W2.append(tf.Variable(tf.random_uniform([hidden_layer_dim,output_dim],-a/sqrt(hidden_layer_dim),a/sqrt(hidden_layer_dim))))
#         b2.append(tf.Variable(tf.random_uniform([output_dim],-a/sqrt(hidden_layer_dim),a/sqrt(hidden_layer_dim))))

	W3.append(tf.Variable(tf.random_uniform([hidden_layer_dim,hidden_layer_dim],-b/sqrt(hidden_layer_dim),b/sqrt(hidden_layer_dim))))
	b3.append(tf.Variable(tf.random_uniform([hidden_layer_dim],-b/sqrt(hidden_layer_dim),b/sqrt(hidden_layer_dim))))

	W4.append(tf.Variable(tf.random_uniform([hidden_layer_dim,hidden_layer_dim],-b/sqrt(hidden_layer_dim),b/sqrt(hidden_layer_dim))))
	b4.append(tf.Variable(tf.random_uniform([hidden_layer_dim],-b/sqrt(hidden_layer_dim),b/sqrt(hidden_layer_dim))))
	W5.append(tf.Variable(tf.random_uniform([hidden_layer_dim,hidden_layer_dim],-b/sqrt(hidden_layer_dim),b/sqrt(hidden_layer_dim))))
	b5.append(tf.Variable(tf.random_uniform([hidden_layer_dim],-b/sqrt(hidden_layer_dim),b/sqrt(hidden_layer_dim))))
	W6.append(tf.Variable(tf.random_uniform([hidden_layer_dim,hidden_layer_dim],-b/sqrt(hidden_layer_dim),b/sqrt(hidden_layer_dim))))
	b6.append(tf.Variable(tf.random_uniform([hidden_layer_dim],-b/sqrt(hidden_layer_dim),b/sqrt(hidden_layer_dim))))
	W7.append(tf.Variable(tf.random_uniform([hidden_layer_dim,hidden_layer_dim],-b/sqrt(hidden_layer_dim),b/sqrt(hidden_layer_dim))))
	b7.append(tf.Variable(tf.random_uniform([hidden_layer_dim],-b/sqrt(hidden_layer_dim),b/sqrt(hidden_layer_dim))))
	W8.append(tf.Variable(tf.random_uniform([hidden_layer_dim,hidden_layer_dim],-b/sqrt(hidden_layer_dim),b/sqrt(hidden_layer_dim))))
	b8.append(tf.Variable(tf.random_uniform([hidden_layer_dim],-b/sqrt(hidden_layer_dim),b/sqrt(hidden_layer_dim))))
	W9.append(tf.Variable(tf.random_uniform([hidden_layer_dim,output_dim],-b/sqrt(hidden_layer_dim),b/sqrt(hidden_layer_dim))))
	b9.append(tf.Variable(tf.random_uniform([output_dim],-b/sqrt(hidden_layer_dim),b/sqrt(hidden_layer_dim))))

#         W1.append(tf.Variable(tf.random_normal([input_dim,hidden_layer_dim],stddev=1/sqrt(input_dim))))
#         b1.append(tf.Variable(tf.random_normal([hidden_layer_dim],stddev=1/sqrt(input_dim))))

#         W2.append(tf.Variable(tf.random_normal([hidden_layer_dim,hidden_layer2_dim],stddev=1/sqrt(hidden_layer_dim))))
#         b2.append(tf.Variable(tf.random_normal([hidden_layer2_dim],stddev=1/sqrt(hidden_layer_dim))))

#         W3.append(tf.Variable(tf.random_normal([hidden_layer2_dim,output_dim],stddev=1/sqrt(hidden_layer2_dim))))
#         b3.append(tf.Variable(tf.random_normal([output_dim],stddev=1/sqrt(hidden_layer2_dim))))

        variables.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name))

x = tf.placeholder(tf.float32, shape=[None, input_dim], name='x')

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
    h3 = tf.matmul(h2, W3[i]) + b3[i]
    h3 = tf.nn.relu(h3)
    h4 = tf.matmul(h3, W4[i]) + b4[i]
    h4 = tf.nn.relu(h4)
    h5 = tf.matmul(h4, W5[i]) + b5[i]
    h4 = tf.nn.relu(h5)
    h6 = tf.matmul(h5, W6[i]) + b6[i]
    h6 = tf.nn.relu(h6)
    h7 = tf.matmul(h6, W7[i]) + b7[i]
    h7 = tf.nn.relu(h7)
    h8 = tf.matmul(h7, W8[i]) + b8[i]
    h8 = tf.nn.relu(h8)
    logits = tf.matmul(h8, W9[i]) + b9[i]
#     logits = tf.matmul(h, W2[i]) + b2[i]
#     logits = tf.matmul(h, W2[i])
    o = tf.sign(logits)
#     outputs.append((o+1)/2)
    outputs.append(tf.reduce_join(tf.reduce_join(tf.as_string(tf.cast((o+1)//2,tf.int32)), 0),0))

session = tf.Session()

inputs = [[float(xx) for xx in "{0:07b}".format(i)] for i in range(2**7)]

# N=10
cnt = Counter()
#weights = {}
for i in range(N):
    # if i%(N/100) == 0:
    print(i)
    session.run(tf.global_variables_initializer())
    fs = session.run(outputs, feed_dict={x:inputs})
    varss = session.run(variables,feed_dict={x:inputs})
    #for i,f in enumerate(fs):
    #    if f in weights:
    #        weights[f].append(varss[i])
    #    else:
    #        weights[f]=[varss[i]]
    cnt += Counter(fs)

pickle.dump(cnt, open( str(idx)+"_cnt_"+str(N*paral_nets)+"_7_"+str(hidden_layer_dim)+"x8_1_allinputs_relu.p", "wb" ), -1)
#pickle.dump(weights, open( str(idx)+"_weights_"+str(N*paral_nets)+"_7_20_20_1_allinputs_relu.p", "wb" ), -1)

#with open(str(idx)+"_comp_freq_7_20_20_1_relu", "w") as f:
#    for fun,val in cnt.most_common():
#        f.write(str(calc_KC(str(fun)))+"\t"+str(val)+"\n")

