
import pickle
from collections import Counter
cnt=Counter()
from KC_LZ import calc_KC
import sys

(idx1,idx2,N) = [int(x) for x in sys.argv[1:]]

if N>0:
	with open("cnt_"+str(N)+"_7_20_20_1_relu.txt", "r") as f:
		for line in f:
			linearr=line[:-1].split("\t")
			linearr[1]=int(linearr[1])
			cnt[linearr[0]]=linearr[1]

for i in range(idx1,idx2):
	print(i)
	cnt += pickle.load(open("7_40_40_1_extra/"+str(i)+"_cnt_250000_7_20_20_1_allinputs_relu.p","rb"))
	N += 250000

with open("cnt_"+str(N)+"_7_40_40_1_relu.txt","w") as f:
	for fun,val in cnt.most_common():
		f.write(fun+"\t"+str(val)+"\n")

