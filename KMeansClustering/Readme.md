This program implements the K-Means Clustering algorithm on the Yelp Dataset using the features latitude, longitude, reviewCounts, and checkins. It also plots the resulting clusters if the option is specified and the within-cluster sum of squared error score. 

To run the program:
`python3 kmeans.py yelp3.csv 12 1 no`

This will run the program with the 4 attributes, 12 clusters, and no plot. 12 Clusters provides the best performance on the standard 4 attributes. 

Since it can take a while to converge, there are print statements to let you know that the program is running.
