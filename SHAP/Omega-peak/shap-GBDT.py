import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import shap
import random
import pandas as pd
import math

natom=10000

title="tanaka-"#"gr-ajliu-"#"q00-1-"#

dumpa=title+"cg-mean-std-1.dat"#title+"00-1-1.dat"#title+"800.1.dat"#title+"1.dat"#
dumpb=title+"cg-mean-std-451.dat"#title+"00-1-451.dat"##title+"800.451.dat"#title+"451.dat"#

dime=36

data={}
data[f'Cu']=1
data[f'Zr']=1
#data[f'raw']=1
for i in range(0,17):
	#j=round(i* 0.2 + 2, 2)
	j=round((i-0.5)* 0.5 + 2, 2)
	data[f'Mean_{j}Å']=[j]
for i in range(0,17):
	#j=round(i* 0.2 + 2, 2)
	j=round((i-0.5)* 0.5 + 2, 2)
	data[f'Std_{j}Å']=[j]

xx=pd.DataFrame(data)
#xx=xx.rename(columns={"1.6Å":"Cu", "1.8Å":"Zr"})
print(xx.columns)
feat_names=list(xx.columns)

pointa = np.zeros((2*natom,dime))

pkl_filename = "pickle_cuzr_"+title+"GBDT-phop-type-acg.pkl"#title + "pickle_model_alpha_cuzr_dgx_alpha-type.pkl"#"pickle_cuzr_"+title+"GBDT.pkl"#"pickle_model_cuzr_qr-svm-alpha.pkl" #
with open(pkl_filename, 'rb') as file:
	model = pickle.load(file)
file.close

with open(title+"scale-param-GBDT-phop-type-acg.dat",'rb') as fa:#(title+"scale-param-dgx-alpha-ridge.dat",'r') as fa:#("scale-param-qr-alpha.dat",'r') as fa:#
	me = np.zeros((dime))
	st = np.zeros((dime))
	for i in range(0,dime,1):
		line=fa.readline().split()
		me[i] = float(line[0])
		st[i] = float(line[1])
fa.close

#=====================================================================
#=====================================================================

#with open("tanaka-00-1-1.dat",'r') as f:
#	for i in range(0,natom,1):
#		line=f.readline().split()
#		for j in range(0,2,1):
#			pointa[i,j]=float(line[j])
#			pointa[i+natom,j]=float(line[j])
#f.close

with open(dumpa,'r') as f:
	for i in range(0,natom,1):
		line=f.readline().split()
		#print(len(line))
		for j in range(0,dime,1):
			pointa[i,j]=float(line[j])
f.close

with open(dumpb,'r') as f:
	for i in range(0,natom,1):
		line=f.readline().split()
		for j in range(0,dime,1):
			pointa[natom+i,j]=float(line[j])
f.close


#=====================================================================
#=====================================================================

length=2000
cul=0
ocupy=np.zeros((2*natom))
x_test=np.zeros((length,dime))
xxx=np.zeros((length,dime))

for i in range(0,natom,1):
	dice=random.randint(0, 2*natom-1)
	if(ocupy[dice]==0):
		for j in range(0,dime,1):
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
#explainer_gbdt = shap.TreeExplainer(model)
#shap_values_gbdt = explainer_gbdt.shap_values(x_test)

#svm_model = model.named_steps['svc']
best_model = model#.best_estimator_

explainer = shap.TreeExplainer(best_model, x_test)
shap_values_gbdt = explainer.shap_values(x_test)

average_shap_values = np.abs(shap_values_gbdt).mean(axis=0)

with open("shap-"+title+"gbdt-phop-type-acg.dat",'w') as f:
	for i in range(0,dime,1):
		f.write(str(average_shap_values[i])+"\n")
f.close


xxxx = pd.DataFrame(xxx, columns=feat_names)

feature_name = "Mean_3.25Å"
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
df.to_csv("cuzr_"+str(title)+"-m3.25.csv", sep=" ", index=False)
 
# SHAP可视化
shap.initjs()

#shap.dependence_plot("Zr_4.2Å", shap_values_gbdt, xxxx, interaction_index="Cu")

shap.summary_plot(shap_values_gbdt, plot_type="bar")

# GBDT可视化
shap.summary_plot(shap_values_gbdt, x_test, feature_names=feat_names, max_display=10, plot_size=(5, 6))
