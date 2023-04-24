#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Machine Learning for Data Science
#Name = 'Kia Aalaei' Project = 'Binary Rice Classification' Dataset = 'https://www.kaggle.com/datasets/seymasa/rice-dataset-gonenjasmine'
#Base model logistic regression - Noisy Indicators


# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy.random as random


# In[2]:


import seaborn as sns


# In[3]:


rice = sns.load_dataset('RiceGonenandJasmine')


# In[6]:


#Features is x and lables are y 

X = rice.drop(['id','Class'], axis = 1)
X["Discrete"] = 120
X["Continous"] = [random.uniform(100,200) for _ in range(len(X))]


# In[7]:


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




