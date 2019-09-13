# K-Means Clustering

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
dataset = dataset[4:5]

# Using the elbow method to find the optimal number of clusters
set.seed(6)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(dataset, i)$withinss)
# The Elbow method is a heuristic method of interpretation and validation of consistency within cluster analysis designed 
# to help finding the appropriate number of clusters in a dataset.

# More precisely, if one plots the percentage of variance explained by the clusters against the number of clusters, 
# the first clusters will add much information (explain a lot of variance), but at some point the marginal gain will drop, 
# giving an angle in the graph. The number of clusters is chosen at this point, hence the "elbow criterion".

plot(1:10,
     wcss,
     type = 'b',
     main = paste('The Elbow Method'),
     xlab = 'Number of clusters',
     ylab = 'WCSS')

# Fitting K-Means to the dataset
set.seed(0)
kmeans = kmeans(x = dataset, centers = 5)
y_kmeans = kmeans$cluster

cat("Fitting K-Means to the dataset:", y_kmeans)

# Visualising the clusters
library(cluster)
clusplot(dataset,
         y_kmeans,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')