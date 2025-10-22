import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import shap
import random
import pandas as pd
import math

# Function to handle outliers, 异常值索引找到
def handle_shap(df, length):
	in_index = []
	Q1 = np.percentile(df[:], 25)       # 下四分位
	Q3 = np.percentile(df[:], 75)       # 上四分位
	IQR = Q3 - Q1                          # IQR
	lower_bound = Q1 - 1.5 * IQR           # 下边缘
	upper_bound = Q3 + 1.5 * IQR           # 上边缘
	# 寻找异常点, 获得异常点索引值, 删除索引值所在行数据
	for ii in range(0, length):
		if ( df[ii] >= lower_bound and df[ii] <= upper_bound ):
			in_index.append(df[ii])
	return in_index

natom=10000

title="gx-"#"gr-ajliu-"#"tanaka-"#"type-"#"q00-1-"#

#dumpa=title+"800.1.dat"#title+"cg-mean-std-1.dat"#title+"00-1-1.dat"#title+"cg-mean-std-1.dat"#title+"1.dat"#
#dumpb=title+"800.451.dat"#title+"cg-mean-std-451.dat"#title+"00-1-451.dat"#title+"cg-mean-std-451.dat"#title+"451.dat"#
titlea="gr-ajliu-"#"q00-1-"#"type-"#
titlec="tanaka-"
dumpc=titlec+"cg-mean-std-1.dat"
dumpa=titlea+"800.1.dat"#
dumpb=titlea+"800.451.dat"#

dime=80#+34
nn=2

data={}
data[f'Cu']=1
data[f'Zr']=1
#data[f'raw']=1
#for i in range(0,17):
#	j=round((i-0.5)* 0.5 + 2, 2)
#	data[f'Mean_{j}Å']=[j]
#for i in range(0,17):
#	j=round((i-0.5)* 0.5 + 2, 2)
#	data[f'Std_{j}Å']=[j]
for i in range(0,40):
	j=round(i* 0.2 + 2, 2)
	#j=round((i-0.5)* 0.5 + 2, 2)
	data[f'Cu_{j}Å']=[j]
for i in range(0,40):
	j=round(i* 0.2 + 2, 2)
	#j=round((i-0.5)* 0.5 + 2, 2)
	data[f'Zr_{j}Å']=[j]


xx=pd.DataFrame(data)
#xx=xx.rename(columns={"1.6Å":"Cu", "1.8Å":"Zr"})
print(xx.columns)
feat_names=list(xx.columns)

pointa = np.zeros((2*natom,dime+nn))

pkl_filename = "pickle_cuzr_"+title+"GBDT-phop-type-cv.pkl"#"pickle_model_cuzr_gr-alpha.pkl"#title+"pickle_model_alpha_cuzr_dgx_alpha.pkl"#
with open(pkl_filename, 'rb') as file:
	model = pickle.load(file)
file.close

with open(title+"scale-param-GBDT-phop-type-cv.dat") as fa:#("scale-param-gr-alpha.dat") as fa:#
	st = np.zeros((dime+nn))
	me = np.zeros((dime+nn))
	for i in range(0,dime+nn,1):
		line=fa.readline().split()
		me[i] = float(line[0])
		st[i] = float(line[1])
fa.close

#=====================================================================
#=====================================================================

#with open("tanaka-00-1-1.dat",'r') as f:
#	for i in range(0,natom,1):
#		line=f.readline().split()
#		for j in range(0,nn,1):
#			pointa[i,j]=float(line[j])
#			pointa[i+natom,j]=float(line[j])
#f.close

with open(dumpc,'r') as f:
	for i in range(0,natom,1):
		line=f.readline().split()
		#print(len(line))
		for j in range(0,nn,1):
			pointa[i,j]=float(line[j])
f.close

with open(dumpa,'r') as f:
	for i in range(0,natom,1):
		line=f.readline().split()
		#print(len(line))
		for j in range(0,80,1):
			pointa[i,j+nn]=float(line[j])
f.close

with open(dumpb,'r') as f:
	for i in range(0,natom,1):
		line=f.readline().split()
		for j in range(0,dime,1):
			pointa[natom+i,j+nn]=float(line[j])
f.close


#=====================================================================
#=====================================================================

length=2000
cul=0
ocupy=np.zeros((natom))
x_test=np.zeros((length,dime+nn))
xxx=np.zeros((length,dime+nn))

for i in range(0,natom,1):
	dice=random.randint(0, natom-1)
	if(ocupy[dice]==0):
		for j in range(0,dime+nn,1):
			xxx[cul,j] = pointa[dice,j]
			if st[j] == 0:  # 避免除以零
				x_test[cul,j] = 0
			else:
				x_test[cul,j] = (pointa[dice,j] - me[j])/st[j]
		ocupy[dice]=1
		cul += 1
	if (cul==length):
		break
		
# 获取SHAP值

best_model = model.best_estimator_

#all_params = best_model.get_params()
#print("最佳模型的所有参数：", all_params)

#svm_model = best_model.named_steps['svc']
explainer = shap.TreeExplainer(best_model, x_test)#shap.LinearExplainer(svm_model, x_test)
shap_values_gbdt = explainer.shap_values(x_test)
#

print(shap_values_gbdt.shape)
average_shap_values=np.zeros((dime+nn))

for ii in range(0,dime+nn):

	shap_test=np.zeros((length))
	shap_test[:] = shap_values_gbdt[:,ii]

	in_index = handle_shap(shap_test, length)
	
	average_shap_values[ii] = np.abs(in_index).mean()#(axis=0)[in_index]

	#print(average_shap_values.shape)

with open("shap-"+title+"bgdt-phopgx-nonetype.dat",'w') as f:
	for i in range(0,dime+nn,1):
		f.write(str(average_shap_values[i])+"\n")
f.close


xxxx = pd.DataFrame(xxx, columns=feat_names)

feature_name = "Zr_4.2Å"
interaction_feature = "Cu"

# 获取目标特征的 SHAP 值
shap_values_target = shap_values_gbdt[:, xxxx.columns.get_loc(feature_name)]

# 获取目标特征的原始值
feature_values = xxxx[feature_name].values

# 获取交互特征的值
interaction_values = xxxx[interaction_feature].values

# 整理数据
df = pd.DataFrame({
    feature_name: feature_values,
    "SHAP Value": shap_values_target,
    interaction_feature: interaction_values
})

# 输出数据
df.to_csv("cuzr_"+str(title)+"-zr4.2.csv", sep=" ", index=False)
 
# SHAP可视化
shap.initjs()

#shap.dependence_plot("Zr_4.2Å", shap_values_gbdt, xxxx, interaction_index="Cu")

shap.summary_plot(shap_values_gbdt, plot_type="bar")

# GBDT可视化
shap.summary_plot(shap_values_gbdt, x_test, feature_names=feat_names, max_display=10, plot_size=(5, 6))

