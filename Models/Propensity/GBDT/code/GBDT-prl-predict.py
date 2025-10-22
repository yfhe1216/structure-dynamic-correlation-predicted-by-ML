import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib
import matplotlib.pyplot as plt
import pickle
import math
import gc

natom=10000

dime=34#41#
nn=2

title_list = ['voronoivolume', 'u', 'anisotropy', 'tanakac', 'q6', 'f5'] # 

for title in title_list:

	print(title)
	dumpd=title+"-"

	pkl_filename = "pickle_cuzr_"+title+"-acg-alpha-type-cv.pkl"
	with open(pkl_filename, 'rb') as file:
		model = pickle.load(file)
	file.close

	with open(title+"-mean-std-acg-alpha.dat",'r') as fa:
		me = np.zeros((dime+nn))
		st = np.zeros((dime+nn))
		for i in range(0,dime+nn,1):
			line=fa.readline().split()
			me[i] = float(line[0])
			st[i] = float(line[1])
	fa.close

#=====================================================================
#=====================================================================

	for ii in range(1,500,50):
	
		pointb = np.zeros((natom,dime+nn))
		X_test_std=np.zeros((natom,dime+nn))
		point=np.zeros((natom))
	
		dumptc=title+"-cg-mean-std-"+str(ii)+".dat"#title+"-00-1-"+str(ii)+".dat"#
		with open(dumptc,'r') as f:
			for i in range(0,natom,1):
				line=f.readline().split()
				for j in range(0,dime+nn,1):
					pointb[i,j]=float(line[j])
		f.close
	
		for i in range(0,dime+nn,1):
			if st[i] == 0:
				X_test_std[:, i] = 0
			else:
				X_test_std[:,i] = (pointb[:,i] - me[i])/st[i]
	
		y_predict = model.predict(X_test_std)
	
		for j in range(0,natom,1):
			point[j]=float(y_predict[j])

		with open("result-acg-"+dumpd+str(ii)+"-alpha-type.dat", 'w') as gg:
			for j in range(0,natom,1):
				pts = point[j]
				my_string = str(pts).replace('[', '').replace(']', '')
				gg.write(str(my_string)+"\n")
		gg.close

	del me, st, pointb, X_test_std, point
	gc.collect() 

	