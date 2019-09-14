# Apriori

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

dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)

# Creating a sparse matrix using the CSV file
## Removing duplicated transactions
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)

# Plotting the item frequency (N = 100)
itemFrequencyPlot(dataset, topN = 100)

# Training Apriori on the dataset

## Mining Associations with Apriori: 
### Mine frequent itemsets, association rules or association hyperedges using the Apriori algorithm. 
### The Apriori algorithm employs level-wise search for frequent itemsets. 
### The implementation of Apriori used includes some improvements (e.g., a prefix tree and item sorting).
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Apriori - Algorithm
## Step 1: Set a mininum support and confidence
## Step 2: Take all the subsets in transactions having higher support than minimum support
## Step 3: Take all the rules of these subsets having higher confidence than minimum confidence
## Step 4: Sort the rules by decreasing lift

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])

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

# Analyzes
# [01] people how buy {light cream} ================================> also by {chicken} ========> in 29.06% (0.2905983) of the cases
# [02] people how buy {pasta} ======================================> also by {escalope} =======> in 37.29% (0.3728814) of the cases
# [03] people how buy {pasta} ======================================> also by {shrimp} =========> in 32.20% (0.3220339) of the cases
# [04] people how buy {eggs,ground beef} ===========================> also by {herb & pepper} ==> in 20.67% (0.2066667) of the cases
# [05] people how buy {whole wheat pasta} ==========================> also by {olive oil} ======> in 27.15% (0.2714932) of the cases
# [06] people how buy {herb & pepper,spaghetti} ====================> also by {ground beef} ====> in 39.34% (0.3934426) of the cases
# [07] people how buy {herb & pepper,mineral water} ================> also by {ground beef} ====> in 39.06% (0.3906250) of the cases
# [08] people how buy {tomato sauce} ===============================> also by {ground beef} ====> in 37.74% (0.3773585) of the cases
# [09] people how buy {mushroom cream sauce} =======================> also by {escalope} =======> in 30.07% (0.3006993) of the cases
# [10] people how buy {frozen vegetables,mineral water,spaghetti} ==> also by {ground beef} ====> in 36.67% (0.3666667) of the cases