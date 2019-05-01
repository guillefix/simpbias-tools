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
hidden_layer_dim=20
hidden_layer2_dim=20

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

for i in range(paral_nets):
    scope_name = "net"+str(i)
    with tf.variable_scope(scope_name):
#        W1.append(tf.Variable(tf.random_uniform([input_dim,hidden_layer_dim],-a/sqrt(input_dim),a/sqrt(input_dim))))
#        b1.append(tf.Variable(tf.random_uniform([hidden_layer_dim],-a/sqrt(input_dim),a/sqrt(input_dim))))
#
#        W2.append(tf.Variable(tf.random_uniform([hidden_layer_dim,hidden_layer2_dim],-a/sqrt(hidden_layer_dim),a/sqrt(hidden_layer_dim))))
#        b2.append(tf.Variable(tf.random_uniform([hidden_layer2_dim],-a/sqrt(hidden_layer_dim),a/sqrt(hidden_layer_dim))))

#         W2.append(tf.Variable(tf.random_uniform([hidden_layer_dim,output_dim],-a/sqrt(hidden_layer_dim),a/sqrt(hidden_layer_dim))))
#         b2.append(tf.Variable(tf.random_uniform([output_dim],-a/sqrt(hidden_layer_dim),a/sqrt(hidden_layer_dim))))

#        W3.append(tf.Variable(tf.random_uniform([hidden_layer2_dim,output_dim],-b/sqrt(hidden_layer2_dim),b/sqrt(hidden_layer2_dim))))
#        b3.append(tf.Variable(tf.random_uniform([output_dim],-b/sqrt(hidden_layer2_dim),b/sqrt(hidden_layer2_dim))))

         W1.append(tf.Variable(tf.random_normal([input_dim,hidden_layer_dim],stddev=1/sqrt(input_dim))))
         b1.append(tf.Variable(tf.random_normal([hidden_layer_dim],stddev=0.1)))

         W2.append(tf.Variable(tf.random_normal([hidden_layer_dim,hidden_layer2_dim],stddev=1/sqrt(hidden_layer_dim))))
         b2.append(tf.Variable(tf.random_normal([hidden_layer2_dim],stddev=0.1)))

         W3.append(tf.Variable(tf.random_normal([hidden_layer2_dim,output_dim],stddev=sqrt(2)/sqrt(hidden_layer2_dim))))
         b3.append(tf.Variable(tf.random_normal([output_dim],stddev=0.1)))

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
    logits = tf.matmul(h2, W3[i]) + b3[i]
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

pickle.dump(cnt, open( str(idx)+"_cnt_"+str(N*paral_nets)+"_7_20_20_1_allinputs_relu.p", "wb" ), -1)
#pickle.dump(weights, open( str(idx)+"_weights_"+str(N*paral_nets)+"_7_20_20_1_allinputs_relu.p", "wb" ), -1)

#with open(str(idx)+"_comp_freq_7_20_20_1_relu", "w") as f:
#    for fun,val in cnt.most_common():
#        f.write(str(calc_KC(str(fun)))+"\t"+str(val)+"\n")

