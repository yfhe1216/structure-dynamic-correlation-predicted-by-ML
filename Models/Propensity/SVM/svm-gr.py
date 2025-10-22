#from sklearn import svm
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
#import xgboost as xgb
import pickle

import warnings
warnings.filterwarnings("ignore")

import numpy as np

#nfi=1
natom=5000
dim=80
ndump=1

temp="gr-ajliu-"


#tempp="local-birth-death-persis-entropy.1."
#dumpta=tempp+"1"
#dumptb=tempp+"35"

listx=np.zeros((2*ndump*natom,dim))
listy=np.zeros((2*ndump*natom))

for jj in range(1,ndump+1,1):
	dumpa="fast-alpha-gr.dat"
	with open(dumpa,'r') as f:
		pointa=np.zeros((natom,dim))
		labela=np.zeros((natom))
		for i in range(0,natom,1):
			#print(i)
			labela[i]=1
			line=f.readline().split()
			for j in range(0,dim,1):
				pointa[i,j]=float(line[j])
	f.close

	for i in range(0,natom,1):
		k=(jj-1)*natom
		for j in range(0,dim,1):
			listx[k+i,j]=pointa[i,j]
			listy[k+i]=labela[i]

for jj in range(1,ndump+1,1):
	dumpb="slow-alpha-gr.dat"
	with open(dumpb,'r') as f:
		pointb=np.zeros((natom,dim))
		labelb=np.zeros((natom))
		for i in range(0,natom,1):
			labelb[i]=0
			line=f.readline().split()
			for j in range(0,dim,1):
				pointb[i,j]=float(line[j])
	f.close
	
	for i in range(0,natom,1):
		k=ndump*natom+(jj-1)*natom
		for j in range(0,dim,1):
			listx[k+i,j]=pointb[i,j]
			listy[k+i]=labelb[i]
			
X_train=listx[:,:]
Y_train=listy[:]

print(X_train.shape)
print(Y_train.shape)

#########################################################################

X_train_std=np.zeros((2*ndump*natom,dim))
#X_test_std=np.zeros((natom,6))
with open("scale-param-gr-alpha.dat",'w') as sf:
	for i in range(0,dim,1):
		me = np.mean(X_train[:,i])
		st = np.std(X_train[:,i])
		X_train_std[:,i] = (X_train[:,i] - me)/st
		#X_test_std[:,i] = (X_test[:,i] - me)/st
		sf.write(str(me)+" "+str(st)+"\n")

#########################################################################

svc = SVC(kernel='linear', class_weight='balanced')
model = make_pipeline(svc)

for ii in range(41,42,1):

	print(ii)
	
	fcg=open('C-gamma-gr-alpha.dat','a')
			
	#Xtrain, Xtest, ytrain, ytest = train_test_split(X_train_std, Y_train, random_state=ii)
	
#########################################################################
	
#	dtrain=xgb.DMatrix(Xtrain,label=ytrain)
#	dtest=xgb.DMatrix(Xtest)
#	watchlist = [(dtrain,'train')]
	
#	params={'booster':'gbtree',
#        'objective': 'binary:logistic',
#        'eval_metric': 'auc',
#        'max_depth':5,
#        'lambda':10,
#        'subsample':0.75,
#        'colsample_bytree':0.75,
#        'min_child_weight':2,
#        'eta': 0.025,
#        'seed':0,
#        'nthread':8,
#        'gamma':0.15,
#        'learning_rate' : 0.01}
		
#	bst=xgb.train(params,dtrain,num_boost_round=50,evals=watchlist)
#	ypred=bst.predict(dtest)
	
#	y_pred = (ypred >= 0.5)*1
#	print ('Precesion: %.4f' %metrics.precision_score(ytest,y_pred))
#	print ('Recall: %.4f' % metrics.recall_score(ytest,y_pred))
#	print ('F1-score: %.4f' %metrics.f1_score(ytest,y_pred))
#	print ('Accuracy: %.4f' % metrics.accuracy_score(ytest,y_pred))
#	print ('AUC: %.4f' % metrics.roc_auc_score(ytest,ypred))

#	ypred = bst.predict(dtest)
#	print("测试集每个样本的得分\n",ypred)
#	ypred_leaf = bst.predict(dtest, pred_leaf=True)
#	print("测试集每棵树所属的节点数\n",ypred_leaf)
#	ypred_contribs = bst.predict(dtest, pred_contribs=True)
#	print("特征的重要性\n",ypred_contribs )

#	xgb.plot_importance(bst,height=0.8,title='影响糖尿病的重要特征', ylabel='特征')
#	plt.rc('font', family='Arial Unicode MS', size=14)
#	plt.show()
		
#########################################################################

	from sklearn.model_selection import GridSearchCV

	param_grid = {'svc__C':[1]}#[0.001,0.01,0.1,1,10,100]}
	grid = GridSearchCV(model,param_grid)
	grid.fit(X_train_std, Y_train)
	means = grid.cv_results_['mean_test_score']
	params = grid.cv_results_['params']
	
	for param,mean in zip(means,params):
		fcg.write("%f  with:   %r" % (param,mean)+"\n")
	print(grid.best_params_)
	model=grid.best_estimator_
	
	print("okstop")
	
	pkl_filename = "pickle_model_cuzr_gr-alpha.pkl"
	with open(pkl_filename, 'wb') as file:
		pickle.dump(model, file)

