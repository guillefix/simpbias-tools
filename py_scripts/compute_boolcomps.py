
import pickle
from collections import Counter
cnt=Counter()
from KC_LZ import calc_KC
import sys

(start_index,end_index) = [int(x) for x in sys.argv[1:]]

#for i in range(0,31):
#	cnt += pickle.load(open(str(i)+"_cnt_300000_7_20_20_1_allinputs_relu.p","rb"))

cnt=pickle.load(open("cnt_9000000_comp_freq_7_20_20_1_relu.p", "rb"))

#with open("combined_9000000_comp_freq_7_20_20_1_relu", "w") as f:
#    for fun,val in cnt.most_common():
#        f.write(str(calc_KC(str(fun)))+"\t"+str(val)+"\n")

'''
Boolean expression complexity
'''

from sympy import symbols
from sympy.logic.boolalg import SOPform, POSform
# dontcares = [[float(l) for l in "{0:07b}".format(i)] for i in range(0,2**input_dim) if not (calc_KC("{0:07b}".format(i)) < 10)]
'''
ASSUMES n=7, mainly for the symbols bit
'''
inputs = [[float(l) for l in "{0:07b}".format(i)] for i in range(0,2**7)]
#dontcares = [x for x in full_inputs if x not in inputs]
def bool_complexity(inputs,ttable):
	dontcares = []
	x1,x2,x3,x4,x5,x6,x7=symbols('x1 x2 x3 x4 x5 x6 x7')
	constraints=[inputs[i] for i in range(len(inputs)) if ttable[i] == '1']
	circuit1=SOPform([x1,x2,x3,x4,x5,x6,x7], constraints, dontcares=dontcares)
	circuit2=POSform([x1,x2,x3,x4,x5,x6,x7], constraints, dontcares=dontcares)
	return min(circuit1.count_ops(), circuit2.count_ops())
# inputs[0]
# inputs

#comps=[(bool_complexity(inputs,f),val) for f,val in cnt.most_common()]

with open("combined_"+str(start_index)+"_"+str(end_index)+"_boolcomp_freq_7_20_20_1_relu", "a") as f:
    for fun,val in cnt.most_common()[start_index:end_index]:
        f.write(str(fun)+"\t"+str(bool_complexity(inputs,str(fun)))+"\t"+str(val)+"\n")


