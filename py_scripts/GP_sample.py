import numpy as np
inputs = [np.array([int(x) for x in "{0:07b}".format(i)]) for i in range(2**7)]

def phi(x):
    return max(0,x) # relu
def get_sample(covariance_matrix):
    z = np.random.multivariate_normal(np.zeros(2),covariance_matrix)
    return phi(z[0])*phi(z[1])


def numerator_fun(args):
    z1,z2,covar = args
    return phi(z1)*phi(z2)*np.exp(-0.5*np.dot(np.array([z1,z2]),np.dot(covar.I,np.array([z1,z2])).T))
def denominator_fun(args):
    z1,z2,covar = args
    return np.exp(-0.5*np.dot(np.array([z1,z2]),np.dot(covar.I,np.array([z1,z2])).T))

n=7
sigmaw=np.sqrt(6)
sigmab=0.1

K = np.zeros((len(inputs),len(inputs)))

for i,x in enumerate(inputs):
    for j,y in enumerate(inputs):
        K[i,j] = sigmab**2 + (sigmaw**2 * np.dot(x,y)/n)

num_layers = 2
for l in range(num_layers):
    Kold = np.copy(K) # I see, need to clone it
    K = np.zeros((len(inputs),len(inputs)))
    for i,x in enumerate(inputs):
        #print(i)
        for j,y in list(enumerate(inputs))[i:]:
            costheta = Kold[i,j]/(np.sqrt(Kold[i,i]*Kold[j,j]))
            theta = np.arccos(costheta)
            K[i,j] = sigmab**2 + (sigmaw**2/(2*np.pi))*np.sqrt(Kold[i,i]*Kold[j,j])*(np.sin(theta)+(np.pi-theta)*costheta)
    K = np.maximum(K,K.T)



from pyspark import SparkContext
sc = SparkContext.getOrCreate()
print(sc._jsc.sc().getExecutorMemoryStatus())
print(sc)
print("Ready to go!")

data = sc.textFile(filename)

sample_size=int(1e6)
def get_sample_GP(aa):
    return np.random.multivariate_normal(np.zeros(len(inputs)),K,size=sample_size)

import multiprocessing
pool = multiprocessing.Pool(12)

samples = np.concatenate(pool.map(get_sample_GP,range(10)))
funs=["".join([str(int(np.heaviside(x,1))) for x in sample]) for sample in samples]

pool.close()

#data.take(15)
data = data.map(lambda x: x.split("\t"))
data = data.map(lambda x: "\t".join([x[0],str(calc_KC(x[0])), str(int(x[1]))]))
data.saveAsTextFile("LZ_"+filename)

