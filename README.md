Machine Learning in R
===========================

## Data Preprocessing
1. [Data Preprocessing](#data-preprocessing)
2. [Regression](#regression)
	1. [Simple Linear Regression](#simple-linear-regression)
	2. [Multiple Linear Regression](#multiple-linear-regression)
3. [How to run the R program](#how-to-run-the-r-program)

## Data Preprocessing

1. Data Preprocessing
	* Taking care of missing data
	* Encoding categorical data
	* Splitting the dataset into the Training set and Test set
	* Feature Scaling

	a.  [data_preprocessing.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/1_data_preprocessing/data_preprocessing.r)

## Regression

### Simple Linear Regression

* Importing the dataset (Salary_Data.csv)
* Splitting the dataset into the Training set and Test set
* Fitting Simple Linear Regression to the Training set
* Predicting the Test set results
* Visualising the Training set results (ggplot2: scatter plot)
![Visualising the Training set resultsr](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/1_simple_linear_regression/Visualising-the-Test-set-results.png)
* Visualising the Test set results (ggplot2: scatter plot)
![Visualising the Training set resultsr](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/1_simple_linear_regression/Visualising-the-Test-set-results.png)

a.  [simple_linear_regression.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/1_simple_linear_regression/simple_linear_regression.r)

### Multiple Linear Regression

* Importing the dataset (50_Startups.csv)
* Encoding categorical data
* Splitting the dataset into the Training set and Test set
* Fitting Multiple Linear Regression to the Training set
* Predicting the Test set results

a.  [multiple_linear_regression.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/2_multiple_linear_regression/multiple_linear_regression.r)

b. Multiple Linear Regression - Backward Elimination [backward_elimination.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/2_multiple_linear_regression/backward_elimination.r)

c. Multiple Linear Regression - Automatic Backward Elimination [automatic_backward_elimination.r](https://github.com/ramonfigueiredopessoa/machine_learning_in_r/blob/master/src/2_regression/2_multiple_linear_regression/automatic_backward_elimination.r)

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