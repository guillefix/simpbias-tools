import pandas as pd

d = pd.read_csv("cnt_1e8_7_40_40_1_relu.txt",header=None, delim_whitespace=True,names=["out","freq"],nrows=1e5)
d = pd.read_csv("sorted_KCcnts_834_L20.txt",header=None, delim_whitespace=True,dtype={"out":"str","minIn":"str","maxIn":"str"}, names=["out","minIn","maxIn","Kout","KInMin","KInMax","KInMean","freq"])

#######

d[(d["freq"]<128) & (d["comp3"]<35)]["fun"].iloc[3]

x = d["fun"].iloc[10001]
len(x)
KC_LZ3(x)

for x in d["fun"]:
    KC_LZ3(x)

d["comp1"] = d["out"].map(lambda x: KC_LZ(x))
d["comp1b"] = d["out"].map(lambda x: calc_KC(x))
d["comp2"] = d["fun"].map(lambda x: KC_LZ2(x))
d["comp3"] = d["out"].map(lambda x: KC_LZ3(x))
d["comp3b"] = d["out"].map(lambda x: (KC_LZ3(x)+KC_LZ3(x[::-1]))/2.0)
# (KC_LZ(s)+KC_LZ(s[::-1]))/2.0
import matplotlib.pyplot as plt

%matplotlib

fig, ax = plt.subplots()
# ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=2)

plt.scatter(d["comp1"],d["freq"])
plt.scatter(d["comp1b"],d["freq"])
plt.scatter(d["comp2"],d["freq"])
plt.scatter(d["comp3b"],d["freq"])
plt.hist2d(d["comp3b"]-7,np.log2(d["freq"]), bins=(20,20),weights=np.log2(d["freq"]));
plt.hist2d(d["comp1b"],np.log2(d["freq"]), bins=(20,20),weights=np.log2(d["freq"]));
#plt.yscale("log2")

KC_LZ3(32*"0")

string = "00000000000000000000000000000000000"
# string = x
rounds[round](s,i,l,k)
import numpy as np
def KC_LZ3(string):
    n=len(string)
    s = '0'+string
    c=1
    l=1
    i=0
    k=1
    k_max=1
    stop=0
    round = 0

    rounds = [
              lambda s,i,l,k: (s[i+k] != s[l+k]),
              lambda s,i,l,k: (s[i+k] == s[l+k]),
              lambda s,i,l,k: (i-k<1) or (s[i-k] == s[l+k]),
              lambda s,i,l,k: (i-k<1) or (s[i-k] != s[l+k]),
              lambda s,i,l,k: i+2*k >= n or (s[i+2*k] != s[l+k]),
              lambda s,i,l,k: i+2*k >= n or (s[i+2*k] == s[l+k]),
              lambda s,i,l,k: i-2*k<1 or (s[i-2*k] == s[l+k]),
              lambda s,i,l,k: i-2*k<1 or (s[i-2*k] != s[l+k]),
              ]

    while stop == 0:
        if l+k > n:
            stop = 1
            c = c + np.log2(i+1) + np.log2(k)
        elif rounds[round](s,i,l,k): #if the end of the SB s[i+k] disagrees with the end of the LA s[l+k] then stop this SB
            i += 1 # we increase i after having finished scanning a SB
            if k > k_max: k_max = k # k_max stores the longest LA that has been matched by a SB
            k = 1
            if i == l: # when the past index i reaches the present index, we stop scanning the past
                if round == len(rounds) - 1:
                    #finished all the rounds
                    round = 0
                    c = c + np.log2(i) + np.log2(k_max)
                    l += k_max # we move the "present" index l up by the amount that has been matched
                    k_max = 1 # and we reset max lenght of matched pattern to k.
                else:
                    round += 1
                i = 0
        else:
            k += 1

    return c

def KC_LZ2(string):
    n=len(string)
    s = '0'+string
    c=1
    l=1
    i=0
    k=1
    k_max=1
    stop=0


    # SB search buffer is the part of the past we are scanning
    # LA look-ahead is the part of the string from the present forward being matched to to SB

    while stop==0: #we stop when either a search buffer reaches the end of the string, or when the current look-ahead that is predicted by the longest SB reaches the end of the string
        if l+k>n or s[i+k] != s[l+k]: #if the end of the SB s[i+k] disagrees with the end of the LA s[l+k] then stop this SB
            if k>k_max:
                k_max=k # k_max stores the longest LA that has been matched by a SB

            i=i+1 # we increase i after having finished scanning a SB

            if i==l: # when the index i of possible points in the past reaches the present index, we stop scanning the past. We have scanned a total of l SBs
                #c=c+1 # If we were actually compressing, we would add the new token here. here we just count recounstruction STEPs
                c=c+np.log2(l)+np.log2(k_max)
                l=l+k_max # we move the "present" index l up by the amount that has been matched (in terms of coding, the next k_max bits will be compressed by referencing to the point in the past that predicted it).

                if l+1>n: # if the new l is beyond the ending of the string, then we stop.
                    stop=1

                else: #after STEP,
                    i=0 # we reset the searching index to beginning of SB (beginning of string)
                    k=1 # we reset pattern matching index. Note that we are actually matching against the first bit of the string, because we added an extra 0 above, so i+k is the first bit of the string.
                    k_max=1 # and we reset max lenght of matched pattern to k.
            else:
                k=1 #we've finished matching a pattern in the SB, and we reset the matched pattern length counter.

        else: # I increase k as long as the pattern matches, i.e. as long as s[l+k] bit string can be reconstructed by s[i+k] bit string. Note that the matched pattern can "run over" l because the pattern starts copying itself (see LZ 76 paper). This is just what happens when you apply the cloning tool on photoshop to a region where you've already cloned...
            k=k+1

            # if l+k>n: # if we reach the end of the string while matching, we need to add that to the tokens, and stop.
            #     #c=c+1
            #     c=c+np.log2(l)+np.log2(k)
            #     stop=1



    # a la Lempel and Ziv (IEEE trans inf theory it-22, 75 (1976),
    # h(n)=c(n)/b(n) where c(n) is the kolmogorov complexity
    # and h(n) is a normalised measure of complexity.
    complexity=c;

    #b=n*1.0/np.log2(n)
    #complexity=c/b;

    return complexity


def KC_LZ(string):
    n=len(string)
    s = '0'+string
    c=1
    l=1
    i=0
    k=1
    k_max=1
    stop=0


    # SB search buffer is the part of the past we are scanning
    # LA look-ahead is the part of the string from the present forward being matched to to SB

    while stop==0: #we stop when either a search buffer reaches the end of the string, or when the current look-ahead that is predicted by the longest SB reaches the end of the string
        if s[i+k] != s[l+k]: #if the end of the SB s[i+k] disagrees with the end of the LA s[l+k] then stop this SB
            if k>k_max:
                k_max=k # k_max stores the longest LA that has been matched by a SB

            i=i+1 # we increase i after having finished scanning a SB

            if i==l: # when the index i of possible points in the past reaches the present index, we stop scanning the past. We have scanned a total of l SBs
                c=c+1 # If we were actually compressing, we would add the new token here. here we just count recounstruction STEPs
                # c=c+np.log2(l)
                l=l+k_max # we move the "present" index l up by the amount that has been matched (in terms of coding, the next k_max bits will be compressed by referencing to the point in the past that predicted it).

                if l+1>n: # if the new l is beyond the ending of the string, then we stop.
                    stop=1

                else: #after STEP,
                    i=0 # we reset the searching index to beginning of SB (beginning of string)
                    k=1 # we reset pattern matching index. Note that we are actually matching against the first bit of the string, because we added an extra 0 above, so i+k is the first bit of the string.
                    k_max=1 # and we reset max lenght of matched pattern to k.
            else:
                k=1 #we've finished matching a pattern in the SB, and we reset the matched pattern length counter.

        else: # I increase k as long as the pattern matches, i.e. as long as s[l+k] bit string can be reconstructed by s[i+k] bit string. Note that the matched pattern can "run over" l because the pattern starts copying itself (see LZ 76 paper). This is just what happens when you apply the cloning tool on photoshop to a region where you've already cloned...
            k=k+1

            if l+k>n: # if we reach the end of the string while matching, we need to add that to the tokens, and stop.
                c=c+1
                # c=c+np.log2(l)
                stop=1



    # a la Lempel and Ziv (IEEE trans inf theory it-22, 75 (1976),
    # h(n)=c(n)/b(n) where c(n) is the kolmogorov complexity
    # and h(n) is a normalised measure of complexity.
    complexity=c;

    #b=n*1.0/np.log2(n)
    #complexity=c/b;

    return complexity

def calc_KC(s):
    L = len(s)
    if s == '0'*L or s == '1'*L:
        return np.log2(L)
    else:
        return np.log2(L)*(KC_LZ(s)+KC_LZ(s[::-1]))/2.0


##############


p = np.random.permutation(len(fun))
d["out2"] = d["out"].map(lambda x: "".join([x[i] for i in p]))

freqs = []
comps = []
# n1s = []
# row
ents = []
for row in d.iterrows():
    row = row[1]
    freq = row["freq"]
    comp = calc_KC(row["out2"])
    ents.append(entropy(row["out2"]))
    freqs.append(freq)
    # # n1s.append(len([i for i in fun if i=="1"]))
    comps.append(comp)

fun=d["out"].iloc[0]

inputs = [[float(xx) for xx in "{0:07b}".format(i)] for i in range(2**7)]
inputs2 = [[float(xx) for xx in "{0:07b}".format(i)]/np.sqrt(len([float(xx) for xx in "{0:07b}".format(i) if xx =="1"])) for i in range(1,2**7)]
np.random.shuffle(inputs)
inputs
np.array(inputs).shape
from collections import Counter
N=len(inputs)
N
# X=np.random.choice([-1,1],(N,2))
X = np.array(inputs)
# X.shape
cnt = Counter()
zs=np.random.multivariate_normal(np.zeros(N),np.matmul(X,X.T),int(1e4))
xs = (np.sign(zs)+1)/2
zs.shape
for x in xs:
    s = "".join([str(int(i)) for i in x])
    cnt[s]+=1

plt.matshow(np.matmul(X,X.T))
plt.matshow(np.linalg.inv(np.matmul(X.T,X)))


from complexities import entropy

freqs = []
comps = []
ents = []
# n1s = []
for fun,freq in cnt.most_common():
    # freqs.append(freq)
    ents.append(entropy(fun))
    # n1s.append(len([i for i in fun if i=="1"]))
    # comps.append(calc_KC(fun))
    #comps.append(KC_LZ3(fun))

plt.scatter(comps,freqs)
plt.scatter(ents,freqs)
# plt.hist(n1s,weights=freqs)
plt.plot(freqs)
plt.yscale("log")
plt.xscale("log")
