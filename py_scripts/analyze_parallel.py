
# coding: utf-8

# In[9]:

#import ipyparallel
#c = ipyparallel.Client()
#view = c.load_balanced_view()


# In[10]:

#get_ipython().run_cell_magic('px', '--local', '\nfrom complexities import calc_KC, entropy, hamming_comp,bool_complexity,crit_sample_ratio\n\n# import os\n\n# os.environ["SPARK_HOME"] = "/usr/local/Cellar/apache-spark/1.5.1/"\n# os.environ["PYSPARK_PYTHON"]="/usr/bin/python3"\n# os.environ["PYSPARK_DRIVER_PYTHON"]="/usr/bin/python3"\n\n# os.environ["PYSPARK_PYTHON"]="/usr/local/shared/python/3.6.3/bin/python3"\n# os.environ["PYSPARK_DRIVER_PYTHON"]="/usr/local/shared/python/3.6.3/bin/python3"\n\n\n# import numpy as np\nimport pandas as pd\n# d=pd.read_csv("cnt_51000000_7_20_20_1_relu_no_header.csv", delim_whitespace=True)\n\n# d["HC"] = d[0].map(lambda x: hamming_comp(inputs_str,x,2))\n\n# d.to_csv("cnt_51000000_7_20_20_1_relu.csv", sep="\\t")\n\ninput_dim = 7\ninputs = [[int(l) for l in "{0:07b}".format(i)] for i in range(0,2**input_dim)]\ninputs_str = ["{0:07b}".format(i) for i in range(0,2**input_dim)]')

def compute_comps(x):
	x=x[1]
	return [hamming_comp(inputs_str,x[0],2), entropy(x[0]), calc_KC(x[0]), bool_complexity(inputs,x[0]), crit_sample_ratio(inputs_str,x[0]),int(x[1])]

def test(x):
	return x**2
# In[11]:
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import pandas as pd
if __name__ == '__main__':
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	#from mpi4py_map import map
	if rank == 0:
		print("Reading file")
		d=pd.read_csv("cnt_100000000_7_40_40_1_relu.txt",header=None,delim_whitespace=True)
		print("Finished reading file")
	with open("comp_freq_100000000_7_40_40_1_relu.txt","w") as f:
		with MPIPoolExecutor() as executor:
			print("Sending data to workers")
			#print(list(executor.map(test, range(100))))
			for result in executor.map(compute_comps, d.iterrows(),chunksize=100000,unordered=True):
				f.write("\t".join([str(x) for x in result]))
			#d.to_csv("comp_freq_100000000_7_40_40_1_relu.csv", sep="\t")


# In[3]:

# import os
# os.chdir('/home/guillefix/code/ai/bias/')


# In[12]:

# d
# squared = view.map_sync(compute_func, x)
# view.map_sync(lambda x: x, d.iterrows())


# In[13]:

# list(d.iterrows())[0][1]


# In[8]:

# from complexities import calc_KC, entropy, hamming_comp,bool_complexity,crit_sample_ratio
# input_dim=7
# inputs_str = ["{0:07b}".format(i) for i in range(0,2**input_dim)]
# crit_sample_ratio(inputs_str,list(d.iterrows())[0][1][0])


# In[52]:




# In[ ]:



