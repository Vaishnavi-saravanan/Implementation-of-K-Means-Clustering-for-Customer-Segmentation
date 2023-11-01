# Implementation-of-K-Means-Clustering-for-Customer-Segmentation
# AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

# Equipments Required:
Hardware – PCs

Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm
1.Choose the number of clusters (K): Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

2.Initialize cluster centroids: Randomly select K data points from your dataset as the initial centroids of the clusters.

3.Assign data points to clusters: Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically done using Euclidean distance, but other distance metrics can also be used.

4.Update cluster centroids: Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

5.Repeat steps 3 and 4: Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

6.Evaluate the clustering results: Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

7.Select the best clustering solution: If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements

# Program:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by:Dhanumalya.D
Register Number:212222230030  
```
```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers (1).csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]

for i in range (1,11):
    kmeans=KMeans(n_clusters = i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow matter")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segmets")
```
# Output:
# data.head():
![Screenshot 2023-10-26 092141](https://github.com/Vaishnavi-saravanan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118541897/ad13ebec-3ed8-4855-905b-91ecc6740b52)


# data.info():
![Screenshot 2023-10-26 092148](https://github.com/Vaishnavi-saravanan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118541897/91f11682-7563-46e5-9ff1-9c31bcdb36f5)

# Null Values:
![Screenshot 2023-11-01 212930](https://github.com/Vaishnavi-saravanan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118541897/985156fe-42a1-48b0-86fb-12f1dced9878)


# Elbow Graph:

![Screenshot 2023-10-26 092222](https://github.com/Vaishnavi-saravanan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118541897/004b19b1-61b6-44c2-a9b6-4a31e9e9030a)

# K-Means Cluster Formation:

![Screenshot 2023-11-01 212404](https://github.com/Vaishnavi-saravanan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118541897/bbd08b58-7d8f-4365-95ff-0cd0320d35fa)

# Predicted Value:
![Screenshot 2023-11-01 213451](https://github.com/Vaishnavi-saravanan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118541897/dc0d9a11-a04e-4f0b-9db6-0579e8b37118)

# Final Graph:
![Screenshot 2023-10-26 092312](https://github.com/Vaishnavi-saravanan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118541897/3ecb1549-846e-45b7-a5dd-337bcf7e3081)


# Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
