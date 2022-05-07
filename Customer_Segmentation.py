#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
plt.style.use("fivethirtyeight")


# In[67]:


get_ipython().system('pip install dabl ')
import dabl # for data analysis


# In[68]:


data=pd.read_csv("Mall_Customers.csv")


# In[69]:


data.head()


# In[70]:


print( "Data Shape : " ,data.shape)


# In[71]:


# Sampe of data
data.sample(3)


# In[72]:


# Pairplot for data
sns.pairplot(data)
plt.show()


# In[73]:


# Check correlation with Heat Map
sns.heatmap(data.corr(),annot=True,cmap="copper")
plt.title("Correlation Heatmap",fontsize=12)
plt.show()


# In[74]:


#Analyze the data with respect to spending Score
dabl.plot(data,target_col="Spending Score (1-100)");


# In[75]:


#Analyze the data respect to Annual Income
dabl.plot(data,target_col="Annual Income (k$)");


# In[76]:


#Describing data
data.describe().T


# In[77]:


#Describing the categorical data
data.describe(include="object")


# In[78]:


# checking null data
data.isnull().any().any()


# ##### Data Visualization

# In[79]:


import warnings
warnings.filterwarnings("ignore")


# In[80]:


labels=["Female","Male"]
size=data["Gender"].value_counts()
colors=["green","orange"]
explode=[0,0.001]

plt.rcParams["figure.figsize"]=(9,9)
plt.pie(size,colors=colors,explode=explode,labels=labels,shadow=True,startangle=90,autopct="%.2f%%")
plt.title("Gender Gap",fontsize=20)
plt.axis("off")
plt.legend()
plt.show()


# In[81]:


plt.rcParams["figure.figsize"]=(15,8)
sns.countplot(data["Age"],palette="hsv")
plt.title("Distrubition of Age",fontsize=20)
plt.show()


# In[82]:


plt.figure(figsize=(20,8))
sns.displot(data["Annual Income (k$)"],color="red")
plt.title("Distrubition of Annual Income ",fontsize=20)
plt.show()


# In[83]:


plt.figure(figsize=(20,8))
sns.displot(data["Spending Score (1-100)"],color="black")
plt.title("Distrubition of Spending Score ",fontsize=20)
plt.show();


# In[84]:


# Gender SpendScore
plt.figure(figsize=(20,8))
sns.boxenplot(data["Gender"],data["Spending Score (1-100)"],palette="Blues")
plt.title("Gender vs Spending Score",fontsize=20)
plt.show();


# In[85]:


# Gender vs Annual Income
plt.figure(figsize=(20,8))
sns.violinplot(data["Gender"],data["Annual Income (k$)"],palette="rainbow")
plt.title("Gender vs Annual Income ",fontsize=20)
plt.show();


# In[86]:


# Gender vs Age 
plt.figure(figsize=(20,8))
sns.stripplot(data["Gender"],data["Age"],palette="Purples",size=12)
plt.title("Gender vs Age ",fontsize=20)
plt.show();


# In[87]:


# Annual Income vs Age and Spending Score 
x=data["Annual Income (k$)"]
y=data["Age"]
z=data["Spending Score (1-100)"]


sns.lineplot(x,y,color="blue")
sns.lineplot(x,z,color="pink")
plt.title("Annual Income vs Age and Spending Score")
plt.show()


# In[88]:


x=data.loc[:,["Spending Score (1-100)","Annual Income (k$)"]].values
x.shape


# In[89]:


x_data=pd.DataFrame(x)
x_data.head()


# In[90]:


#Elbow method
from sklearn.cluster import KMeans

wcss=[]
for i in range (1,11):
    km=KMeans(n_clusters=i, init="k-means++",max_iter=300,n_init=10,random_state=42)
    km.fit(x)
    wcss.append(km.inertia_)



plt.plot(range(1,11),wcss)
plt.title("Elbow Method",fontsize=20)
plt.xlabel("Number of Cluster")
plt.ylabel("wcss")
plt.show()


# In[91]:


# we use k=5# 
# lets visualize these clusters
plt.style.use("fivethirtyeight")

km=KMeans(n_clusters=5, init="k-means++",max_iter=300,n_init=10,random_state=42)
y_means=km.fit_predict(x)


plt.scatter(x[y_means==0,0],x[y_means==0,1],s=100,c="pink",label="miser")
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=100,c="yellow",label="general")
plt.scatter(x[y_means==2,0],x[y_means==2,1],s=100,c="cyan",label="target")
plt.scatter(x[y_means==3,0],x[y_means==3,1],s=100,c="magenta",label="spendthrift")
plt.scatter(x[y_means==4,0],x[y_means==4,1],s=100,c="orange",label="careful")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=50,c="blue",label="centroid")



plt.style.use("fivethirtyeight")
plt.title("K Means Clustering between Annual Income ans Spending Score")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.grid()
plt.show()


# In[92]:


# Clustering between Age and Spending Score
from sklearn.cluster import KMeans

wcss=[]
for i in range (1,11):
    km=KMeans(n_clusters=i, init="k-means++",max_iter=300,n_init=10,random_state=42)
    km.fit(x)
    wcss.append(km.inertia_)


plt.rcParams["figure.figsize"]=(15,5)
plt.plot(range(1,11),wcss)
plt.title("K-Means Clustering(Elbow Method)",fontsize=20)
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# In[93]:


kmeans=KMeans(n_clusters=4, init="k-means++",max_iter=300,n_init=10,random_state=42)
ymeans=kmeans.fit_predict(x)

plt.rcParams["figure.figsize"]=(15,15)
plt.title("Cluster of Ages",fontsize=30)


plt.scatter(x[ymeans==0,0],x[ymeans==0,1],s=100,c="pink",label="Usual Customers")
plt.scatter(x[ymeans==1,0],x[ymeans==1,1],s=100,c="orange",label="Priority Customers")
plt.scatter(x[ymeans==2,0],x[ymeans==2,1],s=100,c="green",label="Target Customers(Young)")
plt.scatter(x[ymeans==3,0],x[ymeans==3,1],s=100,c="red",label="Target Customer(old)")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=50,c="black")



plt.style.use("fivethirtyeight")
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.legend()
plt.grid()
plt.show()


# In[ ]:





# In[ ]:




