import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler



title="gr-ajliu-800."

nfi=1
natom=10000
dim=80

pkl_filename = "pickle_model_cuzr_gr-alpha.pkl"
with open(pkl_filename, 'rb') as file:
	model = pickle.load(file)

with open("scale-param-gr-alpha.dat",'r') as fa:
	me = np.zeros((dim))
	st = np.zeros((dim))
	for i in range(0,dim,1):
		line=fa.readline().split()
		me[i] = float(line[0])
		st[i] = float(line[1])
	
print("ok")

for ii in range(1,500,50):#[1,5,9,11,13,15,17,19,20,25,27,29,31,35]:
	
	dumpta=title+str(ii)+".dat"
	pointa = np.zeros((natom,dim))
		
	with open(dumpta,'r') as f:
		for i in range(0,natom,1):
			line=f.readline().split()
			for j in range(0,dim,1):
				pointa[i,j]=float(line[j])

	X_test_std=np.zeros((natom,dim))
	for i in range(0,dim,1):
		X_test_std[:,i] = (pointa[:,i] - me[i])/st[i]

	yfit=model.predict(X_test_std)
	distance=model.decision_function(X_test_std)

	point=np.zeros((natom))
	pointd=np.zeros((natom))
	for j in range(0,natom,1):
		point[j]=float(yfit[j])
		pointd[j]=float(distance[j])

	with open("result-gr-alpha-fs."+str(ii), 'w') as gg:
		for j in range(0,natom,1):
			pts = point[j]
			ptsd = pointd[j]
			my_string = str(pts).replace('[', '').replace(']', '')
			my_stringd = str(ptsd).replace('[', '').replace(']', '')
			gg.write(str(my_string)+" "+str(my_stringd)+"\n")


