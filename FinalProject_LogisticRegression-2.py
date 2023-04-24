#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Machine Learning for Data Science
#Name = 'Kia Aalaei' Project = 'Binary Rice Classification' Dataset = 'https://www.kaggle.com/datasets/seymasa/rice-dataset-gonenjasmine'
#Base model logistic regression - Poly Features


# In[6]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import PolynomialFeatures


# In[7]:


import seaborn as sns


# In[8]:


rice = sns.load_dataset('RiceGonenandJasmine')


# In[9]:


#Features is x and lables are y 

X = rice.drop(['id','Class'], axis = 1)


# In[11]:


y = rice['Class'] 


# In[1]:


poly = PolynomialFeatures(2,3)
poly.fit(X)
Z = poly.transform(X)
print(Z)


# In[13]:


from sklearn.model_selection import train_test_split 
Z_train, Z_test, y_train, y_test = train_test_split( Z, y, test_size = 0.33) 
#67% of dataset to training and 33% to testing 
#Splitting dataset into training and testing


# In[14]:


from sklearn.linear_model import LogisticRegression


# In[15]:


lr = LogisticRegression()


# In[16]:


lr.fit(Z_train,y_train)
score = lr.score(Z_train,y_train)
print("Training score: ", score)


# In[17]:


ypred = lr.predict(Z_test)
confusionmatrix = confusion_matrix(y_test, ypred)
sns.heatmap(confusionmatrix, square=True, annot=True, cbar=False, cmap='YlGnBu') #flag, YlGnBu, jet)
plt.xlabel('predicted value')
plt.ylabel('true value');
print(confusionmatrix)


# In[18]:


cr = classification_report(y_test, ypred)
print(cr)


# In[ ]:




