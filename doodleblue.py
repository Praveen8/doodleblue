#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


train=pd.read_csv('D:\data science\Task\predicter\criminal_train.csv')
test=pd.read_csv('D:\data science\Task\predicter\criminal_test.csv')


# In[5]:


train.head()
test.head()


# In[6]:


train.columns


# In[7]:


train.shape


# In[8]:


train.dtypes


# In[9]:


train.isnull().sum()


# In[10]:


train.describe()


# In[11]:


train.count()


# In[12]:


train.corr()


# In[12]:


#Visualizing the data


# In[13]:


sns.heatmap(train.isnull(),yticklabels=False)


# In[14]:


correlation=train.corr()
sns.heatmap(correlation,cbar=True,square=True,cmap='YlGnBu')


# In[15]:


sns.countplot(x='Criminal',data=train)


# In[13]:


train.drop("PERID",axis=1,inplace=True)


# In[14]:


train.head()


# In[18]:


train.shape


# In[25]:


test.columns


# In[15]:


test.drop("PERID",axis=1,inplace=True)


# In[16]:


test.head()


# In[16]:


X_train = train.drop('Criminal', axis=1)
y_train = train['Criminal']
X_test  = test


# In[18]:


X_train.head()


# In[17]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[20]:


print(X_train)


# In[21]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# In[ ]:


##Predicting the test set result


# In[28]:


y_pred= classifier.predict(X_test)


# In[ ]:





# In[23]:


value_prediction = classifier.predict(X_test)
classifier.score(X_train, y_train)


# In[ ]:


#Modeling


# In[18]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[25]:





# In[ ]:


value_prediction = classifier.predict(X_test)
classifier.score(X_train, y_train)


# In[20]:


classifier=DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train,y_train)


# In[21]:


value_prediction = classifier.predict(X_test)
classifier.score(X_train, y_train)


# In[29]:


pickling = pd.DataFrame({    
    "PERID": test["PERID"],
    "Criminal": y_pred,
    })
pickling.to_csv('pickling.csv', index=False, columns=['PERID', 'Criminal'])


# In[33]:


result = pd.read_csv('pickling.csv')
result.head()

