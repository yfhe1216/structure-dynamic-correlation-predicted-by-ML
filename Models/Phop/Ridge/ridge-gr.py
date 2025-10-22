from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn import metrics
#import xgboost as xgb
import pickle

import warnings
warnings.filterwarnings("ignore")

import numpy as np

#nfi=1
natom=10000
dime=80#40#
nn=2

title="gr-ajliu-"#"q00-1-"#

score=title+"alpha-cuzr-phop-ridge.dat"
pkl_filename = title+"pickle_model_cuzr_phop-ridge.pkl"

dumpta=title+"800.1.dat"#title+"1.dat"#
dumptb=title+"800.451.dat"#title+"451.dat"#

pointa = np.zeros((2*natom,dime))
Y_train = np.zeros((2*natom))
X_train = np.zeros((2*natom))

with open(dumpta,'r') as f:
	for i in range(0,natom,1):
		line=f.readline().split()
		for j in range(0,dime,1):
			pointa[i,j]=float(line[j])
f.close
with open(dumptb,'r') as f:
	for i in range(0,natom,1):
		line=f.readline().split()
		for j in range(0,dime,1):
			pointa[i+natom,j]=float(line[j])
f.close

with open("phop-800K-5ps-1.dump",'r') as f:
	for i in range(0,9,1):
		f.readline()
	for i in range(0,natom,1):
		line=f.readline().split()
		Y_train[i]=float(line[1])
f.close
with open("phop-800K-5ps-451.dump",'r') as f:
	for i in range(0,9,1):
		f.readline()
	for i in range(0,natom,1):
		line=f.readline().split()
		Y_train[i+natom]=float(line[1])
f.close

X_train=pointa[:,:]

print(X_train.shape)
print(Y_train.shape)

#########################################################################

X_train_std=np.zeros((nn*natom,dime))

with open(title+"scale-param-phop-ridge.dat",'w') as sf:
	for i in range(0,dime,1):
		me = np.mean(X_train[:,i])
		st = np.std(X_train[:,i])
		if st == 0:  # 避免除以零
			X_train_std[:, i] = 0
		else:
			X_train_std[:,i] = (X_train[:,i] - me)/st
		sf.write(str(me)+" "+str(st)+"\n")
sf.close

#########################################################################

ridgeaa = Ridge(alpha=1.0)
ridgeaa.fit(X_train_std, Y_train)

fcg=open(score,'a')

#########################################################################

from sklearn.model_selection import GridSearchCV

param_grid = {'alpha':[0.1,1,10,100,1000,10000]}
grid = GridSearchCV(Ridge(),param_grid,cv=5)
grid.fit(X_train_std, Y_train)
means = grid.cv_results_['mean_test_score']
params = grid.cv_results_['params']
	
for param,mean in zip(means,params):
	fcg.write("%f  with:   %r" % (param,mean)+"\n")
print(grid.best_params_)
model=grid.best_estimator_
	
print("okstop")
	
with open(pkl_filename, 'wb') as file:
	pickle.dump(model, file)





