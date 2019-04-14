import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_cfm(cfm, classes, directory):
	df = pd.DataFrame(columns=classes, index=classes, data=cfm)
	df.to_csv(os.path.join(directory,"cfm.csv"))
	plt.figure()
	plt.imshow(cfm, cmap=plt.cm.Blues)
	indices = range(len(cfm))
	plt.xticks(indices, classes, fontsize=2)
	plt.yticks(indices, classes, fontsize=2)
	plt.colorbar()
	plt.xlabel('guess')
	plt.ylabel('fact')
	#for first_index in range(len(cfm)):
	#        for second_index in range(len(cfm[first_index])):
	#                plt.text(first_index, second_index, cfm[first_index][second_index])
	plt.savefig(os.path.join(directory,"cfm.png"),transparent=False,bbox_inches='tight',dpi=500)
