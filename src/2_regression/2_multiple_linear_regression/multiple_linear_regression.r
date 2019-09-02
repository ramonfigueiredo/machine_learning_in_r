# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# Linear regression (lm) is going to take care of feature scaling for us.
# Therefore we don't need to do feature scaling.

# Fitting Multiple Linear Regression to the Training set
# regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
#                data = training_set)
# or
regressor = lm(formula = Profit ~ ., # dot (.) means all independent variables
               data = training_set)

# Note: R take care of dummy variables (State2, State3 and remove State1 to avoid some redundant dependency), 
# so we don't need to implement it as in Python

summary(regressor)

# Call:
#   lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + 
#        State, data = training_set)
# 
# Residuals:
#   Min     1Q Median     3Q    Max 
# -33128  -4865      5   6098  18065 
# 
# Coefficients:
#                   Estimate  Std. Error t value Pr(>|t|)    
# (Intercept)      4.965e+04  7.637e+03   6.501 1.94e-07 ***
# R.D.Spend        7.986e-01  5.604e-02  14.251 6.70e-16 ***        <=== Means R.D.Spend is statistical significance using p-value = 0.05
# Administration  -2.942e-02  5.828e-02  -0.505    0.617    
# Marketing.Spend  3.268e-02  2.127e-02   1.537    0.134    
# State2           1.213e+02  3.751e+03   0.032    0.974    
# State3           2.376e+02  4.127e+03   0.058    0.954    
# ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 9908 on 34 degrees of freedom
# Multiple R-squared:  0.9499,	Adjusted R-squared:  0.9425 
# F-statistic:   129 on 5 and 34 DF,  p-value: < 2.2e-16



# CONCLUSION
# 
# After look in the summary(regressor), we conclude that R&D spend has the only strong effect on the profits.
# So that's very important information for investors because they now know that they shouldn't only be looking at
# the profit itself, but it should also be looking at the R&d spend variable that should be looking at the amount
# spend in R&D to add another criterion in their investment decisions.
# Therefore, this is a great information and basically what it means is that among all the independent variables
# the only strong predictor of the profit is the R&D spend. The rest is absolutely unnecessary.
# So we can rewrite the regressor from lm(formula = Profit ~ ., data = training_set) to lm(formula = Profit ~ R.D.Spend, data = training_set),
# and that will be OK. That will give a similar prediction.


# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)
