# K-Means Clustering

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
dataset = dataset[4:5]

# Using the elbow method to find the optimal number of clusters
set.seed(0)
wcss = vector()
for (i in 1:10) 
    # withinss = Vector of within-cluster sum of squares, one component per cluster
    wcss[i] = sum(kmeans(dataset, i)$withinss)

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
# Cluster 1 has low income and low spending score. A better name for this cluster of clients as "Sensible clients" 
# Cluster 2 has average income and average spending score. A better name for this cluster of clients as "Standard clients" 
# Cluster 3 has high income and high spending score. A better name for this cluster of clients as "Target clients"
# So, cluster 3 is the cluster of clients that would be the main potential target of the mall marketing campaigns
# and it would be very insighful for them all to understand what kind of products are bought by the clients in this cluster 
# Cluster 4 has low income and high spending score. A better name for this cluster of clients as "Careless clients" 
# Cluster 5 has high income and low spending score. A better name for this cluster of clients as "Careful clients" 