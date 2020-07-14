#!/usr/bin/env python
# coding: utf-8

# In[163]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd 
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# from tqdm import tqdm
# tqdm.pandas()
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    


# In[120]:


train=pd.read_csv(r'C:\Users\ASUS\Desktop\Train.csv', parse_dates=['DATE'])
test=pd.read_csv(r'C:\Users\ASUS\Desktop\Test.csv', parse_dates=['DATE'])

#Checking null values
print(train.isnull().sum())
print()
print(test.isnull().sum())


# In[121]:


#Replacing null values with the mean. You can use any statistic like std, variance etc.
train['X_12'].fillna(train['X_12'].mean(),inplace=True)
test['X_12'].fillna(test['X_12'].mean(),inplace=True)


# In[122]:


#Graph: Similarily you can plot for other features and visualize
sns.scatterplot(y='X_1',x='DATE',data=train)
plt.tight_layout() 


# In[123]:


#No Null Values now in both the datasets
print(train.isnull().sum())
print()
print(test.isnull().sum())


# In[179]:


#Modelling
#We'll only use X_1 to X_15 as the features(X) and MULTIPLE _OFFENSE as the label(Y)
df=train.copy()

X_train , X_test, y_train, y_test = train_test_split(df,df, test_size=.25, random_state=11)

x_train_list=df.columns[2:17]
y_train_list=df.columns[-1]
x_test_list=df.columns[2:17]
y_test_list=df.columns[-1]


X_train = X_train[x_train_list].values
y_train = y_train[y_train_list].values

X_test = X_test[x_test_list].values
y_test = y_test[y_test_list].values

#Standardizing the data (Without Standardizing the accuracy was a bit lesser than after standardizing)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)

#Decision Tree Classifier

classifier1 = DecisionTreeClassifier()
classifier1.fit(X_train, y_train)

y_pred = classifier1.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Decision Tree Accuracy:", round(accuracy_score(y_test, y_pred)*100,2), "%")
print("Decision Tree Precision:", round(precision_score(y_test, y_pred)*100,2),'%')



# In[180]:


#Random Forest Classifier

classifier2 = RandomForestClassifier(15)
classifier2.fit(X_train, y_train)

y_pred = classifier2.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Random Forest Accuracy:", round(accuracy_score(y_test, y_pred)*100,2), "%")
print("Random Forest Precision:", round(precision_score(y_test, y_pred)*100,2),'%')


# In[181]:


#SVM

classifier3 = svm.SVC(kernel='rbf', C = 1.5)
classifier3.fit(X_train, y_train)

y_pred = classifier3.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("SVM Accuracy:", round(accuracy_score(y_test, y_pred)*100,2), "%")
print("SVM Precision:", round(precision_score(y_test, y_pred)*100,2),'%')


# In[182]:


#Logistic Regression

classifier4 = LogisticRegression()
classifier4.fit(X_train, y_train)

y_pred = classifier4.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("LR Accuracy:", round(accuracy_score(y_test, y_pred)*100,2), "%")
print("LR Precision:", round(precision_score(y_test, y_pred)*100,2),'%')


# In[183]:


#XGBoost

classifier4 = XGBClassifier(colsample_bytree = 0.9, learning_rate = 0.5,
                max_depth = 5, alpha = 10, n_estimators = 20)
classifier4.fit(X_train, y_train)

y_pred = classifier4.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Xgboost Classifier Accuracy:", round(accuracy_score(y_test, y_pred)*100,2), "%")
print("Xgboost Classifier Precision:", round(precision_score(y_test, y_pred)*100,2),'%')


# In[ ]:




