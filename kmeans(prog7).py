  import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd

x=pd.read_csv("C:/Users/MCA/Desktop/ML Lab Dataset/driver-data.csv")
print(x.info())
x1=x['mean_dist_day'].values
print(x1)
x2=x['mean_over_speed_perc'].values
print(x2)
x=np.array(list(zip(x1,x2))).reshape(len(x1),2)  #zip-assign value in the form of (1,a)(2,b)......so on and reshape is used to arrange the dimen accrdly
print(x.shape)
plt.show()
plt.xlim([0,250]) #limitation of scale in x asis
plt.ylim([0,100])
plt.title('Dataset')
plt.scatter(x1,x2)
plt.show()

import matplotlib.pyplot as plt1
kmeans=KMeans(n_clusters=3)   #assigning no. of clusters i.e 3
kmeans.fit(x)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
plt.title("KMEANS")
plt1.scatter(x[:,0],x[:,1],c=kmeans.labels_,cmap='rainbow')
plt1.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='black')
plt1.show()     