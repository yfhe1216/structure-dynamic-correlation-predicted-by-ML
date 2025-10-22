import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import pickle
import math
from sklearn.model_selection import GridSearchCV
import pandas as pd
import gc

natom=10000
dime=34#41#
nn=2
y_train = np.zeros((2*natom))
y_test = np.zeros((2*natom))

#=====================================================================
#========y label
#=====================================================================

with open("propensity-alpha-800K-43ps-1.dat",'r') as f:
	for i in range(0,natom,1):
		line=f.readline().split()
		y_train[i]=float(line[1])
f.close
with open("propensity-alpha-800K-43ps-451.dat",'r') as f:
	for i in range(0,natom,1):
		line=f.readline().split()
		y_train[i+natom]=float(line[1])
f.close

with open("propensity-alpha-800K-43ps-251.dat",'r') as f:
	for i in range(0,natom,1):
		line=f.readline().split()
		y_test[i]=float(line[1])
f.close
with open("propensity-alpha-800K-43ps-51.dat",'r') as f:
	for i in range(0,natom,1):
		line=f.readline().split()
		y_test[i+natom]=float(line[1])
f.close

#=====================================================================
#========x train
#=====================================================================

title_list = ['tanakac', 'q6', 'f5' 'voronoivolume', 'u', 'anisotropy', 'tanaka'] 

for title in title_list:

	dumpta=title+"-cg-mean-std-1.dat"#"-00-1-1.dat"#
	dumptb=title+"-cg-mean-std-451.dat"#"-00-1-451.dat"#
	dumptc=title+"-cg-mean-std-251.dat"#"-00-1-251.dat"#
	dumptd=title+"-cg-mean-std-51.dat"#"-00-1-51.dat"#

	pointa = np.zeros((2*natom,dime+nn))
	pointb = np.zeros((2*natom,dime+nn))
	X_train_std = np.zeros((2*natom,dime+nn))
	X_test_std=np.zeros((2*natom,dime+nn))
		
	with open(dumpta,'r') as f:
		for i in range(0,natom,1):
			line=f.readline().split()
			for j in range(0,dime+nn,1):
				pointa[i,j]=float(line[j])
	f.close

	with open(dumptb,'r') as f:
		for i in range(0,natom,1):
			line=f.readline().split()
			for j in range(0,dime+nn,1):
				pointa[i+natom,j]=float(line[j])
	f.close
			
	with open(title+"-mean-std-acg-alpha.dat",'w') as sf:
		for i in range(0,dime+nn,1):
			me = np.mean(pointa[:,i])
			st = np.std(pointa[:,i])
			if st == 0:  # avoid zero
				X_train_std[:, i] = 0
			else:
				X_train_std[:,i] = (pointa[:,i] - me)/st
			sf.write(str(me)+" "+str(st)+"\n")
	sf.close

#=====================================================================
#train
#=====================================================================

	GBDTreg = GradientBoostingRegressor()

	# range of optimized hyperparameters in training 
	param_grid = {
		'n_estimators': range(30,90,10), 
		'max_depth': range(3,9,2),  
		'min_samples_split': [5, 10, 20], 
		'learning_rate': [0.01, 0.1, 1], 
	}

	# use GridSearchCV
	grid_search = GridSearchCV(GBDTreg, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=100, n_jobs=6)
	grid_search.fit(X_train_std, y_train)

	#print("Best parameters:", grid_search.best_params_)
	#print("Best neg_mean_squared_error:", grid_search.best_score_)
	
	results = pd.DataFrame(grid_search.cv_results_)

	# rank
	results = results.sort_values(by="mean_test_score", ascending=False)

	# select top 20
	top20 = results.head(20)

	y_pred = grid_search.predict(X_train_std)
	mse  = mean_squared_error(y_train, y_pred)
	rmse_train = np.sqrt(mse)
	
#=====================================================================
#test
#=====================================================================
		
	with open(title+"-mean-std-acg-alpha.dat",'r') as fa:
		me = np.zeros((dime+nn))
		st = np.zeros((dime+nn))
		for i in range(0,dime+nn,1):
			line=fa.readline().split()
			me[i] = float(line[0])
			st[i] = float(line[1])
	fa.close

	with open(dumptc,'r') as f:
		for i in range(0,natom,1):
			line=f.readline().split()
			for j in range(0,dime+nn,1):
				pointb[i,j]=float(line[j])
	f.close
	with open(dumptd,'r') as f:
		for i in range(0,natom,1):
			line=f.readline().split()
			for j in range(0,dime+nn,1):
				pointb[i+natom,j]=float(line[j])
	f.close

	for i in range(0,dime+nn,1):
		if st[i] == 0:  # avoid zero
			X_test_std[:, i] = 0
		else:
			X_test_std[:,i] = (pointb[:,i] - me[i])/st[i]
			
	model=grid_search
	cul=0
	param0 = top20["params"].iloc[cul]
	
	while True:
	# test
		y_pred = model.predict(X_test_std)
		mse  = mean_squared_error(y_test, y_pred)
		rmse_test = np.sqrt(mse)
		print(float(rmse_train)/float(rmse_test))
	
		# overfit or not
		if float(rmse_train)/float(rmse_test) <= 0.80:
			cul += 1
			if cul >= len(top20):
				print("not top20")
				break
			param0 = top20["params"].iloc[cul]
			model = GradientBoostingRegressor(**param0)
			model.fit(X_train_std, y_train)
			continue
		else:
			break

	# save model
	with open("pickle_cuzr_"+title+"-acg-alpha-type-cv.pkl", 'wb') as file:
		pickle.dump(model, file)
	file.close
	with open("para_"+title+"-acg-alpha-type-cv.dat", 'w') as file:
		file.write(str(param0)+"\n")
		file.write(str(rmse_train)+" "+str(rmse_test)+"\n")
	file.close
	
	del pointa, pointb, X_train_std, X_test_std, grid_search, results, top20
	gc.collect() 
