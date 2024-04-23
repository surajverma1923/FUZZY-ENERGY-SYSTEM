import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from matplotlib import pyplot as pp
from sklearn import tree
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_text
from sklearn.tree import export_graphviz
from xgboost import plot_importance
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from io import StringIO  
from IPython.display import Image
from os import system
import pydot  
import pydotplus
import docx
import operator

f= open("D:/Datasets/hpc/paths.txt",'w')


def printArray(ints, len): 
    for i in ints[0 : len]: 
        print(i," ",end="",file=f) 
    print(file=f) 
    
def tree_to_paths2(mytree, feature_names):
    tree_ = mytree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print ("def tree({}):".format(", ".join(feature_names)))
    
    paths=[]
    statement=""
    pathlen=0

    def recurse(node, depth,paths,pathlen,statement):
      
        if depth==tree_.max_depth:
            return
        
        name = feature_name[node]
        if name =="undefined!":
            name= str(np.argmax(tree_.value[node][0]))
            name=dtc2.classes_[np.argmax(tree_.value[node][0])]
        threshold = tree_.threshold[node]
        messagetopath="{}&{}:".format(statement,name)
        
        if(len(paths) > pathlen):  
            paths[pathlen]=messagetopath 
        else: 
            paths.append(messagetopath) 
  
        pathlen = pathlen + 1
        
        if tree_.children_left[node]==tree_.children_right[node]:
            printArray(paths, pathlen)
            
            
        else:
            recurse(tree_.children_left[node], depth + 1,paths,pathlen,"False")
            recurse(tree_.children_right[node], depth + 1,paths,pathlen, "True")
           
            
            return

    recurse(0, 1,paths,pathlen,statement)    
   


path = r'onehot_final.csv'
dataset1 = pd.read_csv(path,sep=',',header=0,low_memory=False,infer_datetime_format=True)


feature_list=['T1','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5','T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','Press_mm_hg','Windspeed','Visibility']

print(dataset1[feature_list])


x=dataset1[dataset1.columns[24:105]]
y=dataset1['APPLIANCE_label']



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) 
model = XGBClassifier()
model.fit(X_train, y_train)
fig, ax = pp.subplots(figsize=(10,10))
plot_importance(model,ax=ax)
pp.savefig(r'xgb_imp.jpg',dpi=300)
pp.show()


dtc=DecisionTreeClassifier()
dtc = dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

print("DTC Accuracy:",metrics.accuracy_score(y_test, y_pred))

dtc_imp=dict(zip(dataset1.columns[24:105], dtc.feature_importances_))
sorted_dtc=sorted(dtc_imp.items(),key=operator.itemgetter(1))



x_labels = [val[0] for val in sorted_dtc if val[1]>0.00001]
y_labels = [round(val[1],4) for val in sorted_dtc if val[1]>0.00001]

pp.figure(figsize=(5, 15))
pp.barh(x_labels,y_labels)
pp.xlabel('DTC Importance Score')
pp.ylabel('Features')
pp.savefig(r'dtc3_imp.jpg',dpi=300,bbox_inches="tight")
pp.show()



xgb_imp=dict(zip(dataset1.columns[24:105], model.feature_importances_))
sorted_xgb=sorted(xgb_imp.items(),key=operator.itemgetter(1))
print(sorted_xgb)
x_labels = [val[0] for val in sorted_xgb if val[1]>0.00001]
y_labels = [round(val[1],4) for val in sorted_xgb if val[1]>0.00001]
pp.figure(figsize=(5, 10))
pp.barh(x_labels,y_labels)
pp.xlabel('XGB Importance Score')
pp.ylabel('Features')
pp.show()

best_features=[]

for val in sorted_xgb:
    if val[1]>0.035:
        best_features.append(val[0])
        
for val in sorted_dtc:
    if val[1]>0.045:
        best_features.append(val[0])

best_features = list(dict.fromkeys(best_features))
print(best_features)
bf= open("D:/Datasets/hpc/bestfeatures.txt",'w')
print(best_features,file=bf)

x=dataset1[best_features]
y=dataset1['APPLIANCE_label']

print(x)
print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
dtc2=DecisionTreeClassifier(random_state=0)
dtc2 = dtc2.fit(X_train,y_train)
y_pred = dtc2.predict(X_test)

print("DTC2 Accuracy:",metrics.accuracy_score(y_test, y_pred))

tree_to_paths2(dtc2,best_features)
f.close()
bf.close()