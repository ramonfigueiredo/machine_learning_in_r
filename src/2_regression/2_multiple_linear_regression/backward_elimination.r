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
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = training_set)

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
#                   Estimate Std. Error   t value Pr(>|t|)    
# (Intercept)      4.965e+04  7.637e+03   6.501 1.94e-07 ***
#   R.D.Spend        7.986e-01  5.604e-02  14.251 6.70e-16 ***
#   Administration  -2.942e-02  5.828e-02  -0.505    0.617    
# Marketing.Spend  3.268e-02  2.127e-02   1.537    0.134    
# State2           1.213e+02  3.751e+03   0.032    0.974          <==== remove
# State3           2.376e+02  4.127e+03   0.058    0.954          <==== remove
# ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 9908 on 34 degrees of freedom
# Multiple R-squared:  0.9499,	Adjusted R-squared:  0.9425 
# F-statistic:   129 on 5 and 34 DF,  p-value: < 2.2e-16

### Removing State variable from the equation (less significate)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = training_set)

summary(regressor)

# Call:
#   lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, 
#      data = training_set)
# 
# Residuals:
#   Min     1Q Median     3Q    Max 
# -33117  -4858    -36   6020  17957 
# 
# Coefficients:
#                   Estimate Std. Error   t value Pr(>|t|)    
# (Intercept)      4.970e+04  7.120e+03   6.980 3.48e-08 ***
#   R.D.Spend        7.983e-01  5.356e-02  14.905  < 2e-16 ***
#   Administration  -2.895e-02  5.603e-02  -0.517    0.609        <==== remove
# Marketing.Spend  3.283e-02  1.987e-02   1.652    0.107          
# ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 9629 on 36 degrees of freedom
# Multiple R-squared:  0.9499,	Adjusted R-squared:  0.9457 
# F-statistic: 227.6 on 3 and 36 DF,  p-value: < 2.2e-16

### Removing Administration variable from the equation (less significate)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = training_set)

summary(regressor)

# Call:
#   lm(formula = Profit ~ R.D.Spend + Marketing.Spend, data = training_set)
# 
# Residuals:
#   Min     1Q Median     3Q    Max 
# -33294  -4763   -354   6351  17693 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)     4.638e+04  3.019e+03  15.364   <2e-16 ***
#   R.D.Spend       7.879e-01  4.916e-02  16.026   <2e-16 ***
#   Marketing.Spend 3.538e-02  1.905e-02   1.857   0.0713 .         <==== remove (dot (.) means somehow significante, but because 0.0713 > 0.05 let's remove Administration)
# ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 9533 on 37 degrees of freedom
# Multiple R-squared:  0.9495,	Adjusted R-squared:  0.9468 
# F-statistic: 348.1 on 2 and 37 DF,  p-value: < 2.2e-16

### Removing Administration variable from the equation (less significate)

regressor = lm(formula = Profit ~ R.D.Spend,
               data = training_set)

summary(regressor)



# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)