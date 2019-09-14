# Eclat

# Data Preprocessing
install.packages('arules')
library(arules)

## The dataset describes a store located in one of the most popular places in the south of France.
## So, a lot of people go into the store.

## And therefore the manager of the store noticed and calculated that on 
## average each customer goes and buys something to the store once a week.

## This dataset contains 7500 transactions of all the different customers
## that bought a basket of products in a whole week.

## Indeed the manage took it as the basis of its analysis because since each customer is going an 
## average once a week to the store then the transaction registered over a week is 
## quite representative of what customers want to buy.

## So based on all these 7500 transactions our machine learning model (apriori) is going to learn the
## the different associations it can make to actually understand the rules.
## Such as if customers buy this product then they're likely to buy this other set of products.

## Each line in the database (each observation) corresponds to a specific customer who bought a specific basket of product.
## For example, in line 2 the customer bought burgers, meatballs, and eggs

dataset = read.csv('Market_Basket_Optimisation.csv')

# Creating a sparse matrix using the CSV file
## Removing duplicated transactions

dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)

# Plotting the item frequency (N = 100)
itemFrequencyPlot(dataset, topN = 100)

# Training Eclat on the dataset

## Mining Associations with Eclat
### Mine frequent itemsets with the Eclat algorithm. 
### This algorithm uses simple intersection operations for equivalence class clustering along with bottom-up lattice traversal.
rules = eclat(data = dataset, parameter = list(support = 0.003, minlen = 2))

# Eclat - Algorithm
## Step 1: Set a minimum support
## Step 2: Take all the subsets in transactions having higher support than minimum support
## Step 3: Sort the rules by decreasing lift

# Visualising the results
inspect(sort(rules, by = 'support')[1:10])

# items                                  support    count
# [1]  {mineral water,spaghetti}         0.05972537 448  
# [2]  {chocolate,mineral water}         0.05265965 395  
# [3]  {eggs,mineral water}              0.05092654 382  
# [4]  {milk,mineral water}              0.04799360 360  
# [5]  {ground beef,mineral water}       0.04092788 307  
# [6]  {ground beef,spaghetti}           0.03919477 294  
# [7]  {chocolate,spaghetti}             0.03919477 294  
# [8]  {eggs,spaghetti}                  0.03652846 274  
# [9]  {eggs,french fries}               0.03639515 273  
# [10] {frozen vegetables,mineral water} 0.03572857 268  

# Analyzes
## Different sets of items most frequently purchased together

# [01] The sets of items most frequently purchased together are {mineral water,spaghetti} with a support of point of 0.05972537
# [02] The sets of items most frequently purchased together are {chocolate,mineral water} with a support of point of 0.05265965  
# [03] The sets of items most frequently purchased together are {eggs,mineral water} with a support of point of 0.05092654
# [04] The sets of items most frequently purchased together are {milk,mineral water} with a support of point of 0.04799360
# [05] The sets of items most frequently purchased together are {ground beef,mineral water} with a support of point of 0.04092788
# [06] The sets of items most frequently purchased together are {ground beef,spaghetti} with a support of point of 0.03919477
# [07] The sets of items most frequently purchased together are {chocolate,spaghetti} with a support of point of  0.03919477
# [08] The sets of items most frequently purchased together are {eggs,spaghetti} with a support of point of 0.03652846
# [09] The sets of items most frequently purchased together are {eggs,french fries} with a support of point of 0.03639515
# [10] The sets of items most frequently purchased together are {frozen vegetables,mineral water} with a support of point of 0.03572857