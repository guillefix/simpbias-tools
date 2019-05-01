
f_out = open("combined_9000000_boolcomp_freq_7_20_20_1_relu","w")

for i in range(0,59):
	with open("combined_"+str(i*100000)+"_"+str((i+1)*100000)+"_boolcomp_freq_7_20_20_1_relu","r") as f:
		for line in f:
			f_out.write(line)
