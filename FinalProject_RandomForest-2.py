#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Machine Learning for Data Science
#Name = 'Kia Aalaei' Project = 'Binary Rice Classification' Dataset = 'https://www.kaggle.com/datasets/seymasa/rice-dataset-gonenjasmine'
#Random Forest - Poly Features


# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import PolynomialFeatures


# In[2]:


import seaborn as sns


# In[3]:


rice = sns.load_dataset('RiceGonenandJasmine')


# In[4]:


#Features is x and lables are y 

X = rice.drop(['id','Class'], axis = 1)


# In[5]:


y = rice['Class'] 


# In[7]:


poly = PolynomialFeatures(2,3)
poly.fit(X)
Z = poly.transform(X)
print(Z)


# In[8]:


from sklearn.model_selection import train_test_split 
Z_train, Z_test, y_train, y_test = train_test_split( Z, y, test_size = 0.33) 
#67% of dataset to training and 33% to testing 
#Splitting dataset into training and testing


# In[9]:


from sklearn.ensemble import RandomForestClassifier


# In[10]:


rf = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)


# In[11]:


rf.fit(Z_train,y_train)
score = rf.score(Z_train,y_train)
print("Training score: ", score)


# In[12]:


ypred = rf.predict(Z_test)
confusionmatrix = confusion_matrix(y_test, ypred)
sns.heatmap(confusionmatrix, square=True, annot=True, cbar=False, cmap='YlGnBu') #flag, YlGnBu, jet)
plt.xlabel('predicted value')
plt.ylabel('true value');
print(confusionmatrix)


# In[13]:


cr = classification_report(y_test, ypred)
print(cr)


# In[ ]:




