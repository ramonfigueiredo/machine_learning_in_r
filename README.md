Machine Learning in R
===========================

## Contents
1. [Data Preprocessing](#data-preprocessing)
2. [Regression](#regression)
	1. [Simple Linear Regression](#simple-linear-regression)
	2. [Multiple Linear Regression](#multiple-linear-regression)
	3. [Polynomial Regression](#polynomial-regression)
	4. [Support Vector Regression](#support-vector-regression)
	5. [Decision Tree Regressor](#decision-tree-regressor)
	6. [Random Forest Regression](#random-forest-regression)
3. [Classification](#classification)
	1. [Logistic Regression](#logistic-regression)
	2. [K-Nearest Neighbors](#k-nearest-neighbors)
	3. [Support Vector Machine](#support-vector-machine)
	4. [Kernel SVM](#kernel-svm)
	5. [Naive Bayes](#naive-bayes)
	6. [Decision Tree Classification](#decision-tree-classification)
	7. [Random Forest Classification](#random-forest-classification)
4. [Clustering](#clustering)
	1. [K-Means Clustering](#k-means-clustering)
	2. [Hierarchical Clustering](#hierarchical-clustering)
5. [Association Rule Learning](#association-rule-learning)
	1. [Apriori](#apriori)
	2. [Eclat](#eclat)
6. [Reinforcement Learning](#reinforcement-learning)
	1. [Upper Confidence Bound](#upper-confidence-bound)
	2. [Thompson Sampling](#thompson-sampling)
7. [Natural Language Processing](#natural-language-processing)
8. [How to run the R program](#how-to-run-the-r-program)

## Data Preprocessing

a.  [data_preprocessing.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/1_data_preprocessing/data_preprocessing.r)

* Taking care of missing data
* Encoding categorical data
* Splitting the dataset into the Training set and Test set
* Feature Scaling

Go to [Contents](#contents)

## Regression

| Regression Model 				  |	Pros 																					 | Cons |
| ------------------------------- |	---------------------------------------------------------------------------------------- | ---- |
| Linear Regression 			  | Works on any size of dataset, gives informations about relevance of features 			 | The Linear Regression Assumptions |
| Polynomial Regression 		  | Works on any size of dataset, works very well on non linear problems 					 | Need to choose the right polynomial degree for a good bias/variance tradeoff |
| Support Vector Regression (SVR) | Easily adaptable, works very well on non linear problems, not biased by outliers 		 | Compulsory to apply feature scaling, not well known, more difficult to understand |
| Decision Tree Regression  	  | Interpretability, no need for feature scaling, works on both linear / nonlinear problems | Poor results on too small datasets, overfitting can easily occur |
| Random Forest Regression 		  | Powerful and accurate, good performance on many problems, including non linear | No interpretability, overfitting can easily occur, need to choose the number of trees |

### Simple Linear Regression

a.  [simple_linear_regression.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/1_simple_linear_regression/simple_linear_regression.r)

* Importing the dataset (Salary_Data.csv)
* Splitting the dataset into the Training set and Test set
* Fitting Simple Linear Regression to the Training set
* Predicting the Test set results
* Visualising the Training and Test set results

* Visualising the Training set results
![Visualising the Training set resultsr](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/1_simple_linear_regression/Visualising-the-Test-set-results.png)
* Visualising the Test set results
![Visualising the Training set resultsr](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/1_simple_linear_regression/Visualising-the-Test-set-results.png)

Go to [Contents](#contents)

### Multiple Linear Regression

a.  [multiple_linear_regression.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/2_multiple_linear_regression/multiple_linear_regression.r)

b. Multiple Linear Regression - Backward Elimination: [backward_elimination.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/2_multiple_linear_regression/backward_elimination.r)

c. Multiple Linear Regression - Automatic Backward Elimination: [automatic_backward_elimination.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/2_multiple_linear_regression/automatic_backward_elimination.r)

* Importing the dataset (50_Startups.csv)
* Encoding categorical data
* Splitting the dataset into the Training set and Test set
* Fitting Multiple Linear Regression to the Training set
* Predicting the Test set results

Go to [Contents](#contents)

### Polynomial Regression

a.  [polynomial_regression.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/3_polynomial_regression/polynomial_regression.r)

* Importing the dataset ([Position_Salaries.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/3_polynomial_regression/Position_Salaries.csv))
* Fitting Linear Regression to the Training set
* Predicting a new result with Linear Regression
* Visualising the Linear Regression results
* Fitting Polynomial Regression to the Training set
* Predicting a new result with Polynomial Regression
* Visualising the Polynomial Regression results (for higher resolution and smoother curve)

* Visualising the Linear Regression results
![Visualising the Linear Regression results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/3_polynomial_regression/Truth_or_Bluff-Linear_Regression.png)
* Visualising the Polynomial Regression results (degree = 2)
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/3_polynomial_regression/Truth_or_Bluff-Polynomial_Regression-degree_2.png)
* Visualising the Polynomial Regression results (degree = 3)
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/3_polynomial_regression/Truth_or_Bluff-Polynomial_Regression-degree_3.png)
* Visualising the Polynomial Regression results (degree = 4)
![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/3_polynomial_regression/Truth_or_Bluff-Polynomial_Regression-degree_4.png)

Go to [Contents](#contents)

### Support Vector Regression

a.  [svr.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/4_support_vector_regression/svr.r)

* Importing the dataset ([Position_Salaries.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/4_support_vector_regression/Position_Salaries.csv))
* Feature Scaling
* Fitting Support Vector Regression (SVR) to the dataset
* Predicting a new result with Support Vector Regression (SVR)
* Visualising the SVR results (for higher resolution and smoother curve)

![Visualising the SVR results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/4_support_vector_regression/Visualising-the-SVR-results.png)

Go to [Contents](#contents)

### Decision Tree Regressor

a.  [decision_tree_regression.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/5_decision_tree_regression/decision_tree_regression.r)

* Importing the dataset ([Position_Salaries.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/5_decision_tree_regression/Position_Salaries.csv))
* Fitting Decision Tree Regression to the dataset
* Predicting a new result with Decision Tree Regression
* Visualising the Decision Tree Regression results (higher resolution)
* Plotting the tree

![Visualising the Decision Tree Regression results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/5_decision_tree_regression/Visualising-the-Decision-Tree-Regression-results.png)

![Plotting the tree](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/5_decision_tree_regression/Plotting-the-tree.png)

Go to [Contents](#contents)

### Random Forest Regression

a.  [random_forest_regression.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/6_random_forest_regression/random_forest_regression.r)

* Importing the dataset ([Position_Salaries.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/6_random_forest_regression/Position_Salaries.csv))
* Fitting Random Forest Regression to the dataset
* Predicting a new result with Random Forest Regression
* Visualising the Random Forest Regression results (higher resolution)

![Visualising the Random Forest Regression results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/6_random_forest_regression/Visualising-the-Random-Forest-Regression-results.png)

Go to [Contents](#contents)

## Classification

### Logistic Regression

a.  [logistic_regression.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/1_logistic_regression/logistic_regression.r)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/1_logistic_regression/Social_Network_Ads.csv))
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Fitting Logistic Regression to the Training set
* Predicting the Test set results with Logistic Regression
* Making the Confusion Matrix
* Visualising the Training set results
* Visualising the Test set results

![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/1_logistic_regression/Visualising-the-Training-set-results.png)

![Visualising-the-Test-set-results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/1_logistic_regression/Visualising-the-Test-set-results.png)

Go to [Contents](#contents)

### K-Nearest Neighbors

a.  [knn.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/2_k_nearest_neighbors/knn.r)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/2_k_nearest_neighbors/Social_Network_Ads.csv))
* Encoding the target feature as factor
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Fitting K-NN to the Training set and Predicting the Test set results
* Making the Confusion Matrix
* Visualising the Training set results
* Visualising the Test set results

![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/2_k_nearest_neighbors/Visualising-the-Training-set-results.png)

![Visualising-the-Test-set-results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/2_k_nearest_neighbors/Visualising-the-Test-set-results.png)

Go to [Contents](#contents)

### Support Vector Machine

a.  [svm.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/3_support_vector_machine/svm.r)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/3_support_vector_machine/Social_Network_Ads.csv))
* Encoding the target feature as factor
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Fitting SVM to the Training set
* Predicting the Test set results
* Making the Confusion Matrix
* Visualising the Training set results
* Visualising the Test set results

![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/3_support_vector_machine/Visualising-the-Training-set-results.png)

![Visualising-the-Test-set-results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/3_support_vector_machine/Visualising-the-Test-set-results.png)

Go to [Contents](#contents)

### Kernel SVM

a.  [kernel_svm.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/4_kernel_svm/kernel_svm.r)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/4_kernel_svm/Social_Network_Ads.csv))
* Encoding the target feature as factor
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Fitting Kernel SVM to the Training set
* Predicting the Test set results
* Making the Confusion Matrix
* Visualising the Training set results
* Visualising the Test set results

![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/4_kernel_svm/Visualising-the-Training-set-results.png)

![Visualising-the-Test-set-results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/4_kernel_svm/Visualising-the-Test-set-results.png)

Go to [Contents](#contents)

### Naive Bayes

a.  [naive_bayes.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/5_naive_bayes/naive_bayes.r)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/5_naive_bayes/Social_Network_Ads.csv))
* Encoding the target feature as factor
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Fitting Naive Bayes to the Training set
* Predicting the Test set results
* Making the Confusion Matrix
* Visualising the Training set results
* Visualising the Test set results

![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/5_naive_bayes/Visualising-the-Training-set-results.png)

![Visualising-the-Test-set-results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/5_naive_bayes/Visualising-the-Test-set-results.png)

Go to [Contents](#contents)

### Decision Tree Classification

a.  [decision_tree_classification.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/6_decision_tree_classification/decision_tree_classification.r)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/6_decision_tree_classification/Social_Network_Ads.csv))
* Encoding the target feature as factor
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Fitting Decision Tree Classification to the Training set
* Predicting the Test set results
* Making the Confusion Matrix
* Visualising the Training set results
* Visualising the Test set results
* Plotting the tree

![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/6_decision_tree_classification/Visualising-the-Training-set-results.png)

![Visualising-the-Test-set-results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/6_decision_tree_classification/Visualising-the-Test-set-results.png)

![Plotting the tree](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/6_decision_tree_classification/Plotting-the-tree.png)

Go to [Contents](#contents)

### Random Forest Classification

a.  [random_forest_classification.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/7_random_forest_classification/random_forest_classification.r)

* Importing the dataset ([Social_Network_Ads.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/7_random_forest_classification/Social_Network_Ads.csv))
* Encoding the target feature as factor
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Fitting Random Forest Classification to the Training set
* Predicting the Test set results
* Making the Confusion Matrix
* Visualising the Training set results
* Visualising the Test set results
* Choosing the number of trees

![Visualising the Training set results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/7_random_forest_classification/Visualising-the-Training-set-results.png)

![Visualising-the-Test-set-results](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/7_random_forest_classification/Visualising-the-Test-set-results.png)

![Choosing the number of trees](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/3_classification/7_random_forest_classification/Choosing-the-number-of-trees.png)

Go to [Contents](#contents)

## Clustering

| Regression Model 				  |	Pros 																								   | Cons |
| ------------------------------- |	------------------------------------------------------------------------------------------------------------- | ---- |
| K-Means 			  			  | Simple to understand, easily adaptable, works well on small or large datasets, fast, efficient and performant | Need to choose the number of clusters |
| Hierarchical Clustering 		  | The optimal number of clusters can be obtained by the model itself, practical visualisation with the dendrogram | Not appropriate for large datasets |

### K-Means Clustering

a.  [kmeans.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/4_clustering/1_k_means_clustering/kmeans.r)

* Importing the dataset ([Mall_Customers.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/4_clustering/1_k_means_clustering/Mall_Customers.csv))
* Using the [Elbow method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)) to find the optimal number of clusters
	* The Elbow method is a heuristic method of interpretation and validation of consistency within cluster analysis designed to help finding the appropriate number of clusters in a dataset
* Plotting the Elbow method
	* The Elbow method uses the [Within-Cluster Sum of Squares (WCSS)](https://en.wikipedia.org/wiki/K-means_clustering) metric = Sum of squared distances of samples to their closest cluster center.
	* According to the Elbow method the best number of cluster in the mall customers dataset ([Mall_Customers.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/4_clustering/1_k_means_clustering/Mall_Customers.csv)) is 5
* Fitting K-Means to the dataset. The fit method returns for each observation which cluster it belongs to.

![The Elbow Method](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/4_clustering/1_k_means_clustering/The-Elbow-Method.png)

* Visualising the clusters
	* Cluster 1 has low income and low spending score. A better name for this cluster of clients as "Sensible clients" 
	* Cluster 2 has average income and average spending score. A better name for this cluster of clients as "Standard clients" 
	* Cluster 3 has high income and high spending score. A better name for this cluster of clients as "Target clients"
		* So, cluster 3 is the cluster of clients that would be the main potential target of the mall marketing campaigns and it would be very insighful for them all to understand what kind of products are bought by the clients in this cluster 
	* Cluster 4 has low income and high spending score. A better name for this cluster of clients as "Careless clients" 
	* Cluster 5 has high income and low spending score. A better name for this cluster of clients as "Careful clients" 

![Clusters of customers](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/4_clustering/1_k_means_clustering/Clusters-of-customers.png)

Go to [Contents](#contents)

### Hierarchical Clustering

a.  [hierarchical_clustering.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/4_clustering/2_hierarchical_clustering/hierarchical_clustering.r)

* Importing the dataset ([Mall_Customers.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/4_clustering/2_hierarchical_clustering/Mall_Customers.csv))
* Using the [dendrogram](https://en.wikipedia.org/wiki/Dendrogram) to find the optimal number of clusters
* Fitting Hierarchical Clustering to the dataset. The fit method returns for each observation which cluster it belongs to.
* Plotting the Dendrogram (euclidean distance and the ward.D linkage criterion)
	* According to the Dendrogram the best number of cluster in the mall customers dataset ([Mall_Customers.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/4_clustering/1_k_means_clustering/Mall_Customers.csv)) is 5

![Dendrogram](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/4_clustering/2_hierarchical_clustering/Dendrogram.png)

![Dendrogram with 5 clusters](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/4_clustering/2_hierarchical_clustering/Dendrogram-Largest-distance-5_clusters.png)

* Visualising the clusters
	* Cluster 1 has low income and high spending score. A better name for this cluster of clients as "Careless clients" 
	* Cluster 2 has low income and low spending score. A better name for this cluster of clients as "Sensible clients"
	* Cluster 3 has average income and average spending score. A better name for this cluster of clients as "Standard clients"
	* Cluster 4 has high income and low spending score. A better name for this cluster of clients as "Careful clients"
	* Cluster 5 has high income and high spending score. A better name for this cluster of clients as "Target clients"
		* Therefore, cluster 5 is the cluster of clients that would be the main potential target of the mall marketing campaigns and it would be very insighful for them all to understand what kind of products are bought by the clients in this cluster 

![Clusters of customers](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/4_clustering/2_hierarchical_clustering/Clusters-of-customers.png)

* Clusters visialization with different distance metrics and different linkage criterion
	* [See clusters of customers with **canberra** distance and 3 different linkage criterion (**average**, **complete**, **single**, and **ward.D**)](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/4_clustering/2_hierarchical_clustering/Clusters-of-customers-canberra-distance)
	* [See clusters of customers with **euclidean** distance and 3 different linkage criterion (**average**, **complete**, **single**, and **ward.D**)](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/4_clustering/2_hierarchical_clustering/Clusters-of-customers-euclidean-distance)
	* [See clusters of customers with **manhattan** distance and 3 different linkage criterion (**average**, **complete**, **single**, and **ward.D**)](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/4_clustering/2_hierarchical_clustering/Clusters-of-customers-manhattan-distance)
	* [See clusters of customers with **maximum** distance and 3 different linkage criterion (**average**, **complete**, **single**, and **ward.D**)](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/4_clustering/2_hierarchical_clustering/Clusters-of-customers-maximum-distance)
	* [See clusters of customers with **minkowski** distance and 3 different linkage criterion (**average**, **complete**, **single**, and **ward.D**)](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/4_clustering/2_hierarchical_clustering/Clusters-of-customers-minkowski-distance)

Go to [Contents](#contents)

## Association Rule Learning

### Apriori

a.  [apriori.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/5_association_rule_learning/1_apriori/apriori.r)

* Importing the dataset ([Market_Basket_Optimisation.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/5_association_rule_learning/1_apriori/Market_Basket_Optimisation.csv))
	* The dataset describes a store located in one of the most popular places in the south of France. So, a lot of people go into the store.
	* And therefore the manager of the store noticed and calculated that on average each customer goes and buys something to the store once a week.
	* This dataset contains 7500 transactions of all the different customers that bought a basket of products in a whole week.
	* Indeed the manage took it as the basis of its analysis because since each customer is going an average once a week to the store then the transaction registered over a week is quite representative of what customers want to buy.
	* So, based on all these 7500 transactions our machine learning model (*apriori*) is going to learn the different associations it can make to actually understand the rules.
	* Such as if customers buy this product then they're likely to buy this other set of products.
	* Each line in the database corresponds to a specific customer who bought a specific basket of product. 
	* For example, in line 2 the customer bought burgers, meatballs, and eggs.
* Creating a sparse matrix using the CSV file
* Removing duplicated transactions
* Summary of the dataset

```
transactions as itemMatrix in sparse format with
 7501 rows (elements/itemsets/transactions) and
 119 columns (items) and a density of 0.03288973 

most frequent items:
mineral water          eggs     spaghetti  french fries     chocolate       (Other) 
         1788          1348          1306          1282          1229         22405 

element (itemset/transaction) length distribution:
sizes
   1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   18   19   20 
1754 1358 1044  816  667  493  391  324  259  139  102   67   40   22   17    4    1    2    1 

   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
  1.000   2.000   3.000   3.914   5.000  20.000 

includes extended item information - examples:
             labels
1           almonds
2 antioxydant juice
3         asparagus
```

* Plotting the item frequency (N = 100)

![Plotting the item frequency](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/5_association_rule_learning/1_apriori/Plotting-the-item-frequency.png)

* Training Apriori on the dataset

```
Apriori

Parameter specification:
 confidence minval smax arem  aval originalSupport maxtime support minlen maxlen target   ext
        0.2    0.1    1 none FALSE            TRUE       5   0.004      1     10  rules FALSE

Algorithmic control:
 filter tree heap memopt load sort verbose
    0.1 TRUE TRUE  FALSE TRUE    2    TRUE

Absolute minimum support count: 30 

set item appearances ...[0 item(s)] done [0.00s].
set transactions ...[119 item(s), 7501 transaction(s)] done [0.00s].
sorting and recoding items ... [114 item(s)] done [0.00s].
creating transaction tree ... done [0.00s].
checking subsets of size 1 2 3 4 done [0.00s].
writing ... [811 rule(s)] done [0.00s].
creating S4 object  ... done [0.00s].
```

* Visualising the results

```
# lhs                                                 rhs             support     confidence lift     count
# [1]  {light cream}                               => {chicken}       0.004532729 0.2905983  4.843951 34   
# [2]  {pasta}                                     => {escalope}      0.005865885 0.3728814  4.700812 44   
# [3]  {pasta}                                     => {shrimp}        0.005065991 0.3220339  4.506672 38   
# [4]  {eggs,ground beef}                          => {herb & pepper} 0.004132782 0.2066667  4.178455 31   
# [5]  {whole wheat pasta}                         => {olive oil}     0.007998933 0.2714932  4.122410 60   
# [6]  {herb & pepper,spaghetti}                   => {ground beef}   0.006399147 0.3934426  4.004360 48   
# [7]  {herb & pepper,mineral water}               => {ground beef}   0.006665778 0.3906250  3.975683 50   
# [8]  {tomato sauce}                              => {ground beef}   0.005332622 0.3773585  3.840659 40   
# [9]  {mushroom cream sauce}                      => {escalope}      0.005732569 0.3006993  3.790833 43   
# [10] {frozen vegetables,mineral water,spaghetti} => {ground beef}   0.004399413 0.3666667  3.731841 33
```

* Analyzes
	* [01] **people** how **buy {light cream}** also **buy {chicken}** in **29.06%** (0.2905983) of the cases
	* [02] **people** how **buy {pasta}** also **buy {escalope}** in **37.29%** (0.3728814) of the cases
	* [03] **people** how **buy {pasta}** also **buy {shrimp}** in **32.20%** (0.3220339) of the cases
	* [04] **people** how **buy {eggs,ground beef}** also **buy {herb & pepper}** in **20.67%** (0.2066667) of the cases
	* [05] **people** how **buy {whole wheat pasta}** also **buy {olive oil}** in **27.15%** (0.2714932) of the cases
	* [06] **people** how **buy {herb & pepper,spaghetti}** also **buy {ground beef}** in **39.34%** (0.3934426) of the cases
	* [07] **people** how **buy {herb & pepper,mineral water}** also **buy {ground beef}** in **39.06%** (0.3906250) of the cases
	* [08] **people** how **buy {tomato sauce}** also **buy {ground beef}** in **37.74%** (0.3773585) of the cases
	* [09] **people** how **buy {mushroom cream sauce}** also **buy {escalope}** in **30.07%** (0.3006993) of the cases
	* [10] **people** how **buy {frozen vegetables,mineral water,spaghetti}** also **buy {ground beef}** in **36.67%** (0.3666667) of the cases

Go to [Contents](#contents)

### Eclat

a.  [eclat.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/5_association_rule_learning/2_eclat/eclat.r)

* Importing the dataset ([Market_Basket_Optimisation.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/5_association_rule_learning/2_eclat/Market_Basket_Optimisation.csv))
	* The dataset describes a store located in one of the most popular places in the south of France. So, a lot of people go into the store.
	* And therefore the manager of the store noticed and calculated that on average each customer goes and buys something to the store once a week.
	* This dataset contains 7500 transactions of all the different customers that bought a basket of products in a whole week.
	* Indeed the manage took it as the basis of its analysis because since each customer is going an average once a week to the store then the transaction registered over a week is quite representative of what customers want to buy.
	* So, based on all these 7500 transactions our machine learning model (*apriori*) is going to learn the different associations it can make to actually understand the rules.
	* Such as if customers buy this product then they're likely to buy this other set of products.
	* Each line in the database corresponds to a specific customer who bought a specific basket of product. 
	* For example, in line 2 the customer bought burgers, meatballs, and eggs.
* Creating a sparse matrix using the CSV file
* Removing duplicated transactions
* Summary of the dataset

```
transactions as itemMatrix in sparse format with
 7501 rows (elements/itemsets/transactions) and
 119 columns (items) and a density of 0.03288973 

most frequent items:
mineral water          eggs     spaghetti  french fries     chocolate       (Other) 
         1788          1348          1306          1282          1229         22405 

element (itemset/transaction) length distribution:
sizes
   1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   18   19   20 
1754 1358 1044  816  667  493  391  324  259  139  102   67   40   22   17    4    1    2    1 

   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
  1.000   2.000   3.000   3.914   5.000  20.000 

includes extended item information - examples:
             labels
1           almonds
2 antioxydant juice
3         asparagus
```

* Plotting the item frequency (N = 100)

![Plotting the item frequency](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/5_association_rule_learning/2_eclat/Plotting-the-item-frequency.png)

* Training Eclat on the dataset

```
Eclat

parameter specification:
 tidLists support minlen maxlen            target   ext
    FALSE   0.003      2     10 frequent itemsets FALSE

algorithmic control:
 sparse sort verbose
      7   -2    TRUE

Absolute minimum support count: 22 

create itemset ... 
set transactions ...[119 item(s), 7501 transaction(s)] done [0.00s].
sorting and recoding items ... [115 item(s)] done [0.00s].
creating sparse bit matrix ... [115 row(s), 7501 column(s)] done [0.00s].
writing  ... [1328 set(s)] done [0.01s].
Creating S4 object  ... done [0.00s].
```

* Visualising the results

```
     items                             support    count
[1]  {mineral water,spaghetti}         0.05972537 448  
[2]  {chocolate,mineral water}         0.05265965 395  
[3]  {eggs,mineral water}              0.05092654 382  
[4]  {milk,mineral water}              0.04799360 360  
[5]  {ground beef,mineral water}       0.04092788 307  
[6]  {ground beef,spaghetti}           0.03919477 294  
[7]  {chocolate,spaghetti}             0.03919477 294  
[8]  {eggs,spaghetti}                  0.03652846 274  
[9]  {eggs,french fries}               0.03639515 273  
[10] {frozen vegetables,mineral water} 0.03572857 268  
```

* Analyzes (different sets of items most frequently purchased together)
	* [01] The sets of items **most frequently purchased together** are **{mineral water,spaghetti}** with a **support** of point of **0.05972537**
	* [02] The sets of items **most frequently purchased together** are **{chocolate,mineral water}** with a **support** of point of **0.05265965**
	* [03] The sets of items **most frequently purchased together** are **{eggs,mineral water}** with a **support** of point of **0.05092654**
	* [04] The sets of items **most frequently purchased together** are **{milk,mineral water}** with a **support** of point of **0.04799360**
	* [05] The sets of items **most frequently purchased together** are **{ground beef,mineral water}** with a **support** of point of **0.04092788**
	* [06] The sets of items **most frequently purchased together** are **{ground beef,spaghetti}** with a **support** of point of **0.03919477**
	* [07] The sets of items **most frequently purchased together** are **{chocolate,spaghetti}** with a **support** of point of  **0.03919477**
	* [08] The sets of items **most frequently purchased together** are **{eggs,spaghetti}** with a **support** of point of **0.03652846**
	* [09] The sets of items **most frequently purchased together** are **{eggs,french fries}** with a **support** of point of **0.03639515**
	* [10] The sets of items **most frequently purchased together** are **{frozen vegetables,mineral water}** with a **support** of point of **0.03572857**

Go to [Contents](#contents)

## Reinforcement Learning

### Upper Confidence Bound

a.  [random_selection.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/random_selection.r)

* Importing the dataset ([Ads_CTR_Optimisation.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/Ads_CTR_Optimisation.csv))
* Implementing Random Selection
* Visualising the results

![Random selection - Histogram of ads selections](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/Random-selection_Histogram-of-ads-selections.png)

b.  [upper_confidence_bound.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/upper_confidence_bound.r)

* Importing the dataset ([Ads_CTR_Optimisation.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/Ads_CTR_Optimisation.csv))
* Implementing UCB
* Visualising the results

![Upper Confidence Bound (UCB) - Histogram of ads selections](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/UCB-Histogram-of-ads-selections.png)

#### UCB algorithm

**Step 1.** At each round n, we consider two numbers for each ad i:

* ![equation 1](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/equation1.gif) - the number of times the ad i was selected up to round n,

* ![equation 2](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/equation2.gif) - the sum of rewards of the ad i up to round n.

**Step 2.** From these two numbers we compute:

* the average reward of ad i up to round n

![equation 3](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/equation3.gif)

* the confidence interval ![equation 4](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/equation4.gif) at round n with

![equation 4](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/equation5.gif)
		
**Step 3.** We select the ad i that has the maximum UCB ![equation 6](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/1_upper_confidence_bound/equation6.gif)

Go to [Contents](#contents)

### Thompson Sampling

a.  [random_selection.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/2_thompson_sampling/random_selection.r)

* Importing the dataset ([Ads_CTR_Optimisation.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/2_thompson_sampling/Ads_CTR_Optimisation.csv))
* Implementing Random Selection
* Visualising the results

![Random selection - Histogram of ads selections](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/2_thompson_sampling/Random-selection_Histogram-of-ads-selections.png)

b.  [thompson_sampling.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/2_thompson_sampling/thompson_sampling.r)

* Importing the dataset ([Ads_CTR_Optimisation.csv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/2_thompson_sampling/Ads_CTR_Optimisation.csv))
* Implementing Thompson Sampling
* Visualising the results

![Thompson Sampling - Histogram of ads selections](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/2_thompson_sampling/Thompson_Sampling-Histogram-of-ads-selections.png)

#### Thompson Sampling algorithm

**Step 1.** At each round n, we consider two numbers for each ad i:

* ![equation 1](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/2_thompson_sampling/equation1.gif) - the number of times the ad i got reward 1 up to round n,

* ![equation 2](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/2_thompson_sampling/equation2.gif) - the number of times the ad i got reward 0 up to round n.

**Step 2.** For each ad i, we take a random draw from the distribution below:

![equation 3](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/2_thompson_sampling/equation3.gif)

**Step 3.** We select the ad that has the highest ![equation 4](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/6_reinforcement_learning/2_thompson_sampling/equation4.gif)

Go to [Contents](#contents)

## Natural Language Processing

a.  [natural_language_processing.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/7_natural_language_processing/natural_language_processing.r)

* Importing the dataset ([Restaurant_Reviews.tsv](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/7_natural_language_processing/Restaurant_Reviews.tsv))
* Cleaning the texts
* Creating the Bag of Words model
* Splitting the dataset into the Training set and Test set
* Fitting Naive Bayes to the Training set
* Predicting the Test set results
* Making the Confusion Matrix
* Calculating metrics using the confusion matrix

### Metrics: 

* **True Positive (TP):** Observation is positive, and is predicted to be positive.
* **False Negative (FN):** Observation is positive, but is predicted negative.
* **True Negative (TN):** Observation is negative, and is predicted to be negative.
* **False Positive (FP):** Observation is negative, but is predicted positive.

#### Classification Rate / Accuracy

Classification Rate or Accuracy is given by the relation:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

However, there are problems with accuracy.  It assumes equal costs for both kinds of errors. A 99% accuracy can be excellent, good, mediocre, poor or terrible depending upon the problem.

#### Recall

Recall can be defined as the ratio of the total number of correctly classified positive examples divide to the total number of positive examples. High Recall indicates the class is correctly recognized (small number of FN).

Recall is given by the relation:

Recall = TP / (TP + FN)

#### Precision

To get the value of precision we divide the total number of correctly classified positive examples by the total number of predicted positive examples. High Precision indicates an example labeled as positive is indeed positive (small number of FP).

Precision is given by the relation:

Precision = TP / (TP + FP)

High recall, low precision: 
This means that most of the positive examples are correctly recognized (low FN) but there are a lot of false positives.

Low recall, high precision: 
This shows that we miss a lot of positive examples (high FN) but those we predict as positive are indeed positive (low FP)

#### F-measure

Since we have two measures (Precision and Recall) it helps to have a measurement that represents both of them. We calculate an F-measure which uses Harmonic Mean in place of Arithmetic Mean as it punishes the extreme values more.

The F-Measure will always be nearer to the smaller value of Precision or Recall.

Fmeasure = (2 * Recall * Precision) / (Recall + Presision)

### Algorithm output

```
Predicting the Test set results
4   9  10  16  17  21  24  33  39  40  41 
  1   1   1   0   0   0   0   0   1   0   0 
 48  56  58  59  61  63  73  76  82  92  93 
  1   1   0   1   0   1   0   0   0   1   0 
 98  99 105 112 113 115 116 122 123 142 150 
  0   1   0   0   1   1   0   0   1   1   0 
152 154 157 158 159 161 169 182 183 184 188 
  1   0   1   1   1   0   0   0   0   0   0 
190 191 193 199 202 203 210 211 217 222 228 
  1   1   0   1   1   0   0   0   1   0   1 
239 240 250 251 255 258 262 264 270 272 276 
  0   1   0   1   1   0   0   1   0   0   0 
287 292 303 306 314 318 326 328 337 344 345 
  0   0   0   0   0   0   0   1   0   0   0 
346 349 351 353 361 363 364 370 375 395 396 
  1   0   0   0   0   0   1   0   1   1   0 
397 399 412 413 415 416 430 433 445 446 453 
  0   1   1   0   0   0   1   1   1   1   1 
456 466 469 470 473 486 495 496 509 519 521 
  0   1   1   1   1   1   0   0   0   0   1 
525 528 531 535 539 545 548 555 560 563 568 
  1   1   1   0   0   0   0   1   0   1   1 
570 574 583 586 591 598 606 613 614 618 625 
  0   1   0   1   1   1   1   0   0   0   1 
628 633 634 639 641 647 648 653 658 668 674 
  1   0   1   0   1   0   1   1   1   1   1 
679 688 694 698 712 715 716 719 730 739 743 
  1   1   1   0   0   1   1   0   1   1   0 
752 759 761 768 780 789 795 807 809 811 817 
  1   1   1   1   1   1   0   0   1   1   0 
818 821 844 848 849 853 855 863 868 874 882 
  1   0   0   1   0   1   1   0   0   1   0 
890 891 892 894 900 905 906 912 915 920 924 
  1   1   1   0   1   0   1   0   0   1   0 
931 935 938 939 941 953 956 965 973 977 983 
  1   0   0   1   0   1   0   0   0   0   0 
985 996 
  0   0 
Levels: 0 1


Confusion Matrix
	y_pred
     0  1
  0 82 18
  1 23 77


True Positive (TP): 82
False Negative (FN): 18
True Negative (TN): 23
False Positive (FP): 77


Accuracy = (TP + TN) / (TP + TN + FP + FN): 0.525000
Recall = TP / (TP + FN): 0.820000
Precision = TP / (TP + FP): 0.515723
Fmeasure = (2 * recall * precision) / (recall + precision): 0.633205
```

## How to run the R program

1. Install [R](https://www.r-project.org/) and [RStudio](https://www.rstudio.com/)
	* This code uses: R version 3.6.1 and RStudio version 1.2.1335

2. Set environment variable for R on Windows/Linux/MacOS

	* Example of path on Windows: C:\Program Files\R\R-3.6.1\bin
	* Executables: R.exe and Rscript.exe

3. Run the program

```sh
cd <folder_name>/

Rscript.exe <name_of_r_program>.r
```

Or open and run the <name_of_r_program>.r file using RStudio

Go to [Contents](#contents)