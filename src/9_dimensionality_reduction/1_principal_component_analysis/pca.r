# PCA

# Importing the dataset
dataset = read.csv('Wine.csv')

# Splitting the dataset into the Training set and Test set
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-14] = scale(training_set[-14])
test_set[-14] = scale(test_set[-14])

# Applying PCA
install.packages('caret')
library(caret)
install.packages('e1071')
library(e1071)
pca = preProcess(x = training_set[-14], method = 'pca', pcaComp = 2)
training_set = predict(pca, training_set)
training_set = training_set[c(2, 3, 1)]
test_set = predict(pca, test_set)
test_set = test_set[c(2, 3, 1)]

# Fitting SVM to the Training set
install.packages('e1071')
library(e1071)
classifier = svm(formula = Customer_Segment ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

print("Predicting the Test set results")
print(y_pred)

# Creating the Confusion Matrix
cm = table(test_set[, 3], y_pred)

print("Confusion Matrix")
print(cm)

# Calculating metrics using the confusion matrix

TP_class1 = cm[1,1]
TP_class2 = cm[2,2]
TP_class3 = cm[3,3]
sprintf("True Positive (TP) of class 1: %d", TP_class1)
sprintf("True Positive (TP) of class 2: %d", TP_class2)
sprintf("True Positive (TP) of class 3: %d", TP_class3)

sum_matrix_values = cm[1,1] + cm[1,2] + cm[1,3] + cm[2,1] + cm[2,2] + cm[2,3] + cm[3,1] + cm[3,2] + cm[3,3]

print("\nACCURACY, PRECISION, RECALL, F1-SCORE FOR CLASS 1")

TN_class1 = cm[2,2] + cm[2,3] + cm[3,2] + cm[3,3]
accuracy_class1 = (TP_class1 + TN_class1) / sum_matrix_values
sprintf("Accuracy (class 1) = TP (class 1) + cm[2,2] + cm[2,3] + cm[3,2] + cm[3,3] / sum_matrix_values: %f", accuracy_class1 )

precision_class1 = TP_class1 / (cm[1,1] + cm[2,1] + cm[3,1])
sprintf("Precision (class 1) = TP (class 1) / (cm[1,1] + cm[2,1] + cm[3,1]): %f", precision_class1 )

recall_class1 = TP_class1 / (cm[1,1] + cm[1,2] + cm[1,3])
sprintf("Recall (class 1) = TP (class 1) / (cm[1,1] + cm[1,2] + cm[1,3]): %f", recall_class1 )

f1_score_class1 = (2 * recall_class1 * precision_class1) / (recall_class1 + precision_class1)
sprintf("F1-Score (class 1) = (2 * recall_class1 * precision_class1) / (recall_class1 + precision_class1): %f", f1_score_class1 )


print("\nnACCURACY, PRECISION, RECALL, F1-SCORE FOR CLASS 2")

TN_class2 = cm[1,1] + cm[1,3] + cm[3,1] + cm[3,3]
accuracy_class2 = (TP_class2 + TN_class2) / sum_matrix_values
sprintf("Accuracy (class 2) = TP (class 2) + cm[1,1] + cm[1,3] + cm[3,1] + cm[3,3] / sum_matrix_values: %f", accuracy_class2 )

precision_class2 = TP_class2 / (cm[1,2] + cm[2,2] + cm[3,2])
sprintf("Precision (class 2) = TP (class 2) / (cm[1,2] + cm[2,2] + cm[3,2]): %f", precision_class2 )

recall_class2 = TP_class2 / (cm[2,1] + cm[2,2] + cm[2,3])
sprintf("Recall (class 2) = TP (class 2) / (cm[2,1] + cm[2,2] + cm[2,3]): %f", recall_class2 )

f1_score_class2 = (2 * recall_class2 * precision_class2) / (recall_class2 + precision_class2)
sprintf("F1-Score (class 2) = (2 * recall_class2 * precision_class2) / (recall_class2 + precision_class2): %f", f1_score_class2 )


print("\nnACCURACY, PRECISION, RECALL, F1-SCORE FOR CLASS 3")

TN_class3 = cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2]
accuracy_class3 = (TP_class3 + TN_class3) / sum_matrix_values
sprintf("Accuracy (class 3) = TP (class 3) + cm[1,1] + cm[1,2] + cm[2,1] + cm[2,2] / sum_matrix_values: %f", accuracy_class3 )

precision_class3 = TP_class3 / (cm[1,3] + cm[2,3] + cm[3,3])
sprintf("Precision (class 3) = TP (class 3) / (cm[1,3] + cm[2,3] + cm[3,3]): %f", precision_class3 )

recall_class3 = TP_class3 / (cm[3,1] + cm[3,2] + cm[3,3])
sprintf("Recall (class 3) = TP (class 3) / (cm[3,1] + cm[3,2] + cm[3,3]): %f", recall_class3 )

f1_score_class3 = (2 * recall_class3 * precision_class3) / (recall_class3 + precision_class3)
sprintf("F1-Score (class 3) = (2 * recall_class3 * precision_class3) / (recall_class3 + precision_class3): %f", f1_score_class3 )

# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'SVM (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))

