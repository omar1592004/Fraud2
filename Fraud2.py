#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  confusion_matrix,accuracy_score
from sklearn.naive_bayes import GaussianNB
#from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
# Data Pre-processing
data=pd.read_csv("frauddatasetpred.csv")
x = data.dropna()
x1 = x[x['fraud'] != 0]
#print(x1.shape)
x2 = x[x['fraud'] != 1]
#print(x2.shape)
x2=x2.iloc[:-825176,:]
dataset=pd.concat([x1,x2])
dataset['fraud'].value_counts()
#dataset.head()


# In[2]:


# Data scaling & splitting
X = dataset.drop('fraud', axis=1)
Y = dataset['fraud']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "Scaler.pkl")


# In[3]:


# KNN
n = math.ceil(math.sqrt(len(y_train)))
knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
#print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Cross Validation scores:",cross_val_score(knn,X_train_scaled, y_train, cv=5))


# In[4]:


# N_B
model = GaussianNB()
model.fit(X_train_scaled, y_train)
y_pred2= model.predict(X_test_scaled)
accuracy2 = accuracy_score(y_pred2, y_test)
print(f"Accuracy: {accuracy2:.2f}")
print("Cross Validation scores:",cross_val_score(model,X_train_scaled, y_train, cv=5))


# In[5]:


# Log_Reg
model2 = LogisticRegression(max_iter=200)
model2.fit(X_train_scaled,y_train)
y_pred4 = model2.predict(X_test_scaled)
accuracy3 = accuracy_score(y_pred4, y_test)
print(f"Accuracy: {accuracy3:.2f}")
print("Cross Validation scores:",cross_val_score(model2,X_train_scaled, y_train, cv=5))


# In[6]:


#s_v_m
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train_scaled, y_train)
y_pred = svm_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)


# In[7]:


#plot
##cm = confusion_matrix(y_test, y_pred)
##sns.heatmap(cm, annot=True, fmt="d", cmap='OrRd', cbar=False)
##plt.xlabel('Predicted')
##plt.ylabel('True')
##plt.title('Confusion Matrix')
##plt.show()


# In[8]:


import joblib
joblib.dump(knn, 'KNN.pkl')

