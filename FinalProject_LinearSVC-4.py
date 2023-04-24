#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Machine Learning for Data Science
#Name = 'Kia Aalaei' Project = 'Binary Rice Classification' Dataset = 'https://www.kaggle.com/datasets/seymasa/rice-dataset-gonenjasmine'
#SVM (Linear SVC) - Preprocessing Features (StandardScaler())


# In[13]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[14]:


import seaborn as sns


# In[15]:


rice = sns.load_dataset('RiceGonenandJasmine')


# In[16]:


#Features is x and lables are y 

X = rice.drop(['id','Class'], axis = 1)
X.head()


# In[17]:


y = rice['Class'] 
y.head()


# In[18]:


from sklearn.model_selection import train_test_split 
X_train, X_valid, y_train, y_valid = train_test_split( X, y, test_size = 0.33) 
#67% of dataset to training and 33% to testing 
#Splitting dataset into training and testing


# In[19]:


# Make the data scaler
from sklearn.preprocessing import StandardScaler


# In[20]:


# scale the X data    
sc = StandardScaler()
sc.fit(X_train) # fit only on train data


# In[21]:


X_train_sc = sc.transform(X_train)


# In[22]:


X_train = pd.DataFrame(sc.transform(X_train), index=X_train.index, columns=X_train.columns)
X_valid = pd.DataFrame(sc.transform(X_valid), index=X_valid.index, columns=X_valid.columns)


# In[23]:


X_train.head()


# In[25]:


#linear SVC model (SVM)
lsvc = LinearSVC(dual = False)
print(lsvc)


# In[26]:


#fitting model on training data
#training score can be checked
lsvc.fit(X_train,y_train)
score = lsvc.score(X_train,y_train)
print("Training score: ", score)


# In[27]:


#can also use validation method to check training score
cv_score = cross_val_score(lsvc, X_train, y_train, cv=10)
print("CV average score: ", cv_score.mean())


# In[30]:


#Now I can predict the test data using model
ypred = lsvc.predict(X_valid)
confusionmatrix = confusion_matrix(y_valid, ypred)
sns.heatmap(confusionmatrix, square=True, annot=True, cbar=False, cmap='YlGnBu') #flag, YlGnBu, jet)
plt.xlabel('predicted value')
plt.ylabel('true value');
print(confusionmatrix)


# In[32]:


cr = classification_report(y_valid, ypred)
print(cr)


# In[ ]:




