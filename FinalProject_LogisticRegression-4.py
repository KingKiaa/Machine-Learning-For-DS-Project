#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Machine Learning for Data Science
#Name = 'Kia Aalaei' Project = 'Binary Rice Classification' Dataset = 'https://www.kaggle.com/datasets/seymasa/rice-dataset-gonenjasmine'
#Base model logistic regression - Preprocessing Features (StandardScaler())


# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[2]:


import seaborn as sns


# In[3]:


rice = sns.load_dataset('RiceGonenandJasmine')


# In[4]:


#Features is x and lables are y 

X = rice.drop(['id','Class'], axis = 1)


# In[5]:


y = rice['Class'] 


# In[6]:


from sklearn.model_selection import train_test_split 
X_train, X_valid, y_train, y_valid = train_test_split( X, y, test_size = 0.33) 
#67% of dataset to training and 33% to testing 
#Splitting dataset into training and testing


# In[7]:


# Make the data scaler
from sklearn.preprocessing import StandardScaler


# In[8]:


# scale the X data    
sc = StandardScaler()
sc.fit(X_train) # fit only on train data


# In[9]:


X_train_sc = sc.transform(X_train)


# In[10]:


X_train = pd.DataFrame(sc.transform(X_train), index=X_train.index, columns=X_train.columns)
X_valid = pd.DataFrame(sc.transform(X_valid), index=X_valid.index, columns=X_valid.columns)


# In[11]:


from sklearn.linear_model import LogisticRegression


# In[12]:


lr = LogisticRegression()


# In[13]:


lr.fit(X_train,y_train)
score = lr.score(X_train,y_train)
print("Training score: ", score)


# In[15]:


ypred = lr.predict(X_valid)
confusionmatrix = confusion_matrix(y_valid, ypred)
sns.heatmap(confusionmatrix, square=True, annot=True, cbar=False, cmap='YlGnBu') #flag, YlGnBu, jet)
plt.xlabel('predicted value')
plt.ylabel('true value');
print(confusionmatrix)


# In[17]:


cr = classification_report(y_valid, ypred)
print(cr)


# In[ ]:




