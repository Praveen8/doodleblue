#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


train=pd.read_csv('D:\data science\Task\predicter\criminal_train.csv')
test=pd.read_csv('D:\data science\Task\predicter\criminal_test.csv')


# In[10]:


train.head()
test.head()


# In[7]:


train.columns


# In[11]:


train.shape


# In[12]:


train.dtypes


# In[14]:


train.isnull().sum()


# In[16]:


train.describe()


# In[18]:


train.count()


# In[19]:


train.corr()


# In[ ]:


#Visualizing the data


# In[20]:


sns.heatmap(train.isnull(),yticklabels=False)


# In[25]:


correlation=train.corr()
sns.heatmap(correlation,cbar=True,square=True,cmap='YlGnBu')


# In[26]:


sns.countplot(x='Criminal',data=train)


# In[28]:


train.drop("PERID",axis=1,inplace=True)


# In[29]:


train.head()


# In[30]:


train.shape


# In[31]:


test.drop("PERID",axis=1,inplace=True)


# In[32]:


test.head()


# In[33]:


X_train = train.drop('Criminal', axis=1)
y_train = train['Criminal']
X_test  = test


# In[34]:


X_train.head()


# In[35]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[39]:


print(X_train)


# In[40]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# In[ ]:


##Predicting the test set result


# In[41]:


y_pred= classifier.predict(X_test)


# In[ ]:





# In[43]:


value_prediction = classifier.predict(X_test)
classifier.score(X_train, y_train)


# In[ ]:


#Modeling


# In[45]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[56]:


classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train,y_train)


# In[ ]:


value_prediction = classifier.predict(X_test)
classifier.score(X_train, y_train)


# In[ ]:


classifier=DecisionTreeClassifier(n_estimators=10)
classifier.fit(X_train,y_train)


# In[ ]:


pickling = pd.DataFrame({    
    "PERID": test["PERID"],
    "Criminal": y_pred,
    })
pickling.to_csv('pickling.csv', index=False, columns=['PERID', 'Criminal'])

