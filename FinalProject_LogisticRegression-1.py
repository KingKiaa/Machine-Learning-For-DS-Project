#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Machine Learning for Data Science
#Name = 'Kia Aalaei' Project = 'Binary Rice Classification' Dataset = 'https://www.kaggle.com/datasets/seymasa/rice-dataset-gonenjasmine'
#Base model logistic regression - Scale Features


# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[3]:


import seaborn as sns


# In[4]:


rice = sns.load_dataset('RiceGonenandJasmine')


# In[5]:


#Features is x and lables are y 

X = rice.drop(['id','Class'], axis = 1)
X["Area"] = X["Area"] + 2
X["MajorAxisLength"] = X["MajorAxisLength"] * 2
X["MinorAxisLength"] = X["MinorAxisLength"] ** 2
X["Eccentricity"] = X["Eccentricity"] + 3
X["ConvexArea"] = X["ConvexArea"] * 3
X["EquivDiameter"] = X["EquivDiameter"] ** 3
X["Extent"] = X["Extent"] + 4
X["Perimeter"] = X["Perimeter"] * 4
X["Roundness"] = X["Roundness"] ** 4
X["AspectRation"] = X["AspectRation"] + 5
rice.head()


# In[6]:


y = rice['Class'] 


# In[8]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.33) 
#67% of dataset to training and 33% to testing 
#Splitting dataset into training and testing


# In[9]:


from sklearn.linear_model import LogisticRegression


# In[10]:


lr = LogisticRegression()


# In[11]:


lr.fit(X_train,y_train)
score = lr.score(X_train,y_train)
print("Training score: ", score)


# In[12]:


ypred = lr.predict(X_test)
confusionmatrix = confusion_matrix(y_test, ypred)
sns.heatmap(confusionmatrix, square=True, annot=True, cbar=False, cmap='YlGnBu') #flag, YlGnBu, jet)
plt.xlabel('predicted value')
plt.ylabel('true value');
print(confusionmatrix)


# In[13]:


cr = classification_report(y_test, ypred)
print(cr)


# In[ ]:




