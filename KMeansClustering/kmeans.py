import pandas as pd
import numpy as np
from IPython.display import display
import random
import sys
import matplotlib.pyplot as plt


csvFileName = sys.argv[1]

yelpData = pd.read_csv(csvFileName)

k = int(sys.argv[2])
clusteringOption = int(sys.argv[3])
plotOption = sys.argv[4]
plotOptionNumber = 0
if plotOption == 'no':
	plotOptionNumber = 0
else:
	plotOptionNumber = int(sys.argv[4])

#use 3% of the data
if clusteringOption == 5:
	yelpData = yelpData.sample(frac=0.03)
	
#take the attributes and make them into a new dataframe to use
clusteringAttributes = yelpData[['latitude', 'longitude', 'reviewCount', 'checkins']].copy()

print(len(clusteringAttributes))
#do the log transofmration
if clusteringOption == 2:
	clusteringAttributes['reviewCount'] = np.log(clusteringAttributes['reviewCount'])
	clusteringAttributes['checkins'] = np.log(clusteringAttributes['checkins'])

#standardize the data
if clusteringOption == 3:
	clusteringAttributes['reviewCount'] = (clusteringAttributes['reviewCount'] - np.mean(clusteringAttributes['reviewCount'])) / np.std(clusteringAttributes['reviewCount'])
	clusteringAttributes['latitude'] = (clusteringAttributes['latitude'] - np.mean(clusteringAttributes['latitude'])) / np.std(clusteringAttributes['latitude'])
	clusteringAttributes['longitude'] = (clusteringAttributes['longitude'] - np.mean(clusteringAttributes['longitude'])) / np.std(clusteringAttributes['longitude'])
	clusteringAttributes['checkins'] = (clusteringAttributes['checkins'] - np.mean(clusteringAttributes['checkins'])) / np.std(clusteringAttributes['checkins'])

#normalize data to test
if clusteringOption == 6:
	clusteringAttributes = (clusteringAttributes - clusteringAttributes.mean()) / (clusteringAttributes.max() - clusteringAttributes.min())

#plot colors
num_clusters = k

#plt.scatter(clusteringAttributes.checkins, clusteringAttributes.reviewCount, c=next(color))

#randomly select k rows from the dataframe
currentCentroids = clusteringAttributes.sample(n = k)
totalCentroids = currentCentroids.copy()
print(currentCentroids)


#Set the value for what K is that we can use to identify
clusteringAttributes['K-Value'] = 0

#function to calculate manhattan distance
def manhattan_distance(x,y):
    return sum(abs(a-b) for a,b in zip(x,y))


#assign each data point to one cluster, by comparing it with all the clusteringPoints 
#and calculating distance
def closest_points(df, clusteringPoints):
	
	# go through all the data points in the dataset
	for index, row in df.iterrows():
		# go through all the rows in the current centroids
		minimumDistance = sys.maxsize
		kCounter = 0
		kValue = 0
		for index2, row2 in currentCentroids.iterrows():
			#calculate the distance for each of them and store it
			distRow = row.values
			#remove the last value for comparing (it is the k-value field we added earlier)
			distRow = distRow[:-1].copy()
			distRow2 = row2.values 
			
			#calculate the euclidean distance using the norm function
			dist = np.linalg.norm(distRow - distRow2)

			#if the option is 4, we should use manhattan distance
			if clusteringOption == 4:
				dist = manhattan_distance(distRow, distRow2)
				
			kCounter = kCounter + 1
			#remember the minimum distance and store the k-value
			if dist < minimumDistance:
				minimumDistance = dist
				kValue = kCounter
		
		#update the data frame with the k-value
		df.set_value(index, 'K-Value', kValue)

	return df
        
        
#recalculate the centroids by calculating the mean of all the examples in each bin
def recalculate_centroids(df, clusteringPoints, k_Value): 
	#go through all of our possible k-values
	for number in range(1, k_Value + 1):
		#grab the rows that have this k-value
		k_valueFrame = df.loc[df['K-Value'] == number]
		#get the average value for each of their 4 attributes and update the clusteringPointsArray
		clusteringPoints.iloc[number - 1, 0] = k_valueFrame["latitude"].mean()
		clusteringPoints.iloc[number - 1, 1] = k_valueFrame["longitude"].mean()
		clusteringPoints.iloc[number - 1, 2] = k_valueFrame["reviewCount"].mean()
		clusteringPoints.iloc[number - 1, 3] = k_valueFrame["checkins"].mean()
		
	return clusteringPoints


def has_converged(oldDF, newDF):
	if oldDF.equals(newDF):
		return True
	else:
		return False
		
oldDF = clusteringAttributes.copy()
newDF = closest_points(clusteringAttributes, currentCentroids)
currentCentroids = recalculate_centroids(clusteringAttributes, currentCentroids, k)
totalCentroids = totalCentroids.append(currentCentroids)
iterationCount = 1
while not has_converged(oldDF, newDF):
	iterationCount = iterationCount + 1
	print("Went through iteration ", iterationCount)
	oldDF = newDF.copy()
	newDF = closest_points(clusteringAttributes, currentCentroids)
	currentCentroids = recalculate_centroids(clusteringAttributes, currentCentroids, k)
	totalCentroids = totalCentroids.append(currentCentroids)
	#print(newDF.head())


#calculate the within-cluster sum of squared error
#go through all points in each cluster and determine the distance
within_cluster_score = 0
for number in range(1, k + 1):
	#grab the rows that have this k-value
	k_valueFrame = newDF.loc[newDF['K-Value'] == number]

	for index2, row2 in k_valueFrame.iterrows():
		#calculate the distance for each of them and store it
		distRow = row2.values
		#remove the last value for comparing (it is the k-value field we added earlier)
		distRow = distRow[:-1].copy()
		distRow2 = currentCentroids.iloc[number - 1].values
		dist = np.linalg.norm(distRow - distRow2)
		if clusteringOption == 4:
			dist = manhattan_distance(distRow, distRow2)
		dist = dist ** 2
		within_cluster_score = within_cluster_score + dist
		
#calculate the separation of the clusters
cluster_separation_score = 0
for number in range(1, k + 1):
	for number2 in range(1, k + 1):
		distRow3 = currentCentroids.iloc[number2 - 1].values
		distRow4 = currentCentroids.iloc[number - 1].values
		cluster_dist = np.linalg.norm(distRow3 - distRow4)
		if clusteringOption == 4:
			cluster_dist = manhattan_distance(distRow3, distRow4)
		cluster_dist = cluster_dist ** 2
		cluster_separation_score = cluster_separation_score + cluster_dist
	
	
print("WC-SSE Score:")
print(within_cluster_score)
print("")

#print("Cluster Separation Score:")
#print(cluster_separation_score)
print("")

#print the centroids
for number in range(1, k + 1):
	print("Centroid", number, "= ") 
	print(currentCentroids.iloc[number - 1])
	
plotRow = 0 
colors = iter(plt.cm.rainbow(np.linspace(0,1,len(totalCentroids))))

#plot latitude, longitude
if plotOptionNumber == 1:
	for index2, row2 in totalCentroids.iterrows():
		plt.scatter(totalCentroids.iloc[plotRow,0], totalCentroids.iloc[plotRow,1], c=next(colors))
		plotRow = plotRow + 1

#plot reviewCount, checkins
if plotOptionNumber == 2:
	for index2, row2 in totalCentroids.iterrows():
		plt.scatter(totalCentroids.iloc[plotRow,2], totalCentroids.iloc[plotRow,3], c=next(colors))
		plotRow = plotRow + 1


plt.show()

