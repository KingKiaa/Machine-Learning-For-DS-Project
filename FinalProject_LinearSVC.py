#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Machine Learning for Data Science
#Name = 'Kia Aalaei' Project = 'Binary Rice Classification' Dataset = 'https://www.kaggle.com/datasets/seymasa/rice-dataset-gonenjasmine'
#SVM (Linear SVC)


# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.svm import LinearSVC
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
X.head()


# In[6]:


y = rice['Class'] 
y.head()


# In[7]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.33) 
#67% of dataset to training and 33% to testing 
#Splitting dataset into training and testing


# In[8]:


#linear SVC model (SVM)
lsvc = LinearSVC(dual = False)
print(lsvc)


# In[9]:


#fitting model on training data
#training score can be checked
lsvc.fit(X_train,y_train)
score = lsvc.score(X_train,y_train)
print("Training score: ", score)


# In[10]:


#can also use validation method to check training score
cv_score = cross_val_score(lsvc, X_train, y_train, cv=10)
print("CV average score: ", cv_score.mean())


# In[11]:


#Now I can predict the test data using model
ypred = lsvc.predict(X_test)
confusionmatrix = confusion_matrix(y_test, ypred)
sns.heatmap(confusionmatrix, square=True, annot=True, cbar=False, cmap='YlGnBu') #flag, YlGnBu, jet)
plt.xlabel('predicted value')
plt.ylabel('true value');
print(confusionmatrix)


# In[12]:


cr = classification_report(y_test, ypred)
print(cr)


# In[ ]:




