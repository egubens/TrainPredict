`TrainPredict` Package
================

## Overview

`TrainPredict` is an R package designed to streamline the process of
logistic regression modeling for binary classification tasks. With
functions for splitting data, training models, making predictions, and
assessing model performance.

## Installation

To install the package directly from GitHub, use the following command:

``` r
# Install from GitHub using devtools
if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools")
}
devtools::install_github("AU-R-Programming/Final_Project_Group_7")
```

## Getting Started

Once installed, load the `TrainPredict` package to start using its
functionality:

``` r
library(TrainPredict)
```

## Example Workflow

Here is a step-by-step workflow that demonstrates the main functions of
the `TrainPredict` package using the built-in `mtcars` dataset.

### 1. Split Dataset into Training and Test Sets

The `train_test_sampling()` function is used to split the dataset into
training and test sets. Here, we use an 75-25 split.

``` r
# Split the dataset into training and test sets (75-25 split).
# We will use variable `am` as the dependent binary variable. 
data("mtcars")
split_data <- train_test_sampling(mtcars, dependent_var = "am", train_prop = 0.75, return_data = TRUE, seed = 123)

# Extract training and test sets.
train_data <- split_data$train
head(train_data)
```

    ##                 mpg cyl  disp  hp drat    wt  qsec vs am gear carb
    ## Datsun 710     22.8   4 108.0  93 3.85 2.320 18.61  1  1    4    1
    ## Volvo 142E     21.4   4 121.0 109 4.11 2.780 18.60  1  1    4    2
    ## Ford Pantera L 15.8   8 351.0 264 4.22 3.170 14.50  0  1    5    4
    ## Mazda RX4 Wag  21.0   6 160.0 110 3.90 2.875 17.02  0  1    4    4
    ## Toyota Corolla 33.9   4  71.1  65 4.22 1.835 19.90  1  1    4    1
    ## Maserati Bora  15.0   8 301.0 335 3.54 3.570 14.60  0  1    5    8

``` r
test_data <- split_data$test
head(test_data)
```

    ##                      mpg cyl  disp  hp drat    wt  qsec vs am gear carb
    ## Hornet Sportabout   18.7   8 360.0 175 3.15 3.440 17.02  0  0    3    2
    ## Merc 230            22.8   4 140.8  95 3.92 3.150 22.90  1  0    4    2
    ## Cadillac Fleetwood  10.4   8 472.0 205 2.93 5.250 17.98  0  0    3    4
    ## Lincoln Continental 10.4   8 460.0 215 3.00 5.424 17.82  0  0    3    4
    ## Dodge Challenger    15.5   8 318.0 150 2.76 3.520 16.87  0  0    3    2
    ## Fiat X1-9           27.3   4  79.0  66 4.08 1.935 18.90  1  1    4    1

``` r
dim(mtcars)
```

    ## [1] 32 11

``` r
dim(train_data)
```

    ## [1] 24 11

``` r
dim(test_data)
```

    ## [1]  8 11

### 2. Train a Logistic Regression Model

We use the `lr()` function to train a logistic regression model using
selected predictors.

``` r
# Train a logistic regression model using 'hp', 'mpg', and 'wt' as predictors of the 'am' binary variable.
model <- lr(am ~ hp + mpg + wt, data = train_data, B=100, alpha=0.05)

# Display model coefficients.
print(model$beta_optimized)
```

    ##                  [,1]
    ## Intercept -32.4885412
    ## hp          0.1103046
    ## mpg         1.8068521
    ## wt         -6.3142139

``` r
# Display confusion matrix.
print(model$confusion_matrix)
```

    ##        Predicted
    ## Actual  TRUE FALSE
    ##   FALSE    1    13
    ##   TRUE     9     1

### 3. Make Predictions on Test Data

Evaluate the model’s performance on the test dataset using the
`predict_test()` function.

``` r
# Make predictions on the test dataset
test_predictions <- predict_test(model = model, new_data = test_data, dependent_variable_col = "am")
```

    ## [1] "Confusion Matrix:"
    ##          Actual
    ## Predicted 0 1
    ##         0 5 0
    ##         1 0 3

``` r
# Display predictions
(test_predictions)
```

    ##                     actual_outcomes predicted_outcomes
    ## Hornet Sportabout                 0                  0
    ## Merc 230                          0                  0
    ## Cadillac Fleetwood                0                  0
    ## Lincoln Continental               0                  0
    ## Dodge Challenger                  0                  0
    ## Fiat X1-9                         1                  1
    ## Porsche 914-2                     1                  1
    ## Lotus Europa                      1                  1

### 4. Make Predictions on New Data

Use the `predict_new()` function to make predictions for new data
points.

``` r
# Create new data points for prediction
new_data <- data.frame(hp = c(225, 110),
                       mpg = c(15.9, 25.6),
                       wt = c(3.485, 1.985))

# Predict using the trained model
new_predictions <- predict_new(data = new_data, model = model, threshold = 0.5)

# Display predictions
print(new_predictions)
```

    ##    hp  mpg    wt predicted
    ## 1 225 15.9 3.485         0
    ## 2 110 25.6 1.985         1

## Key Functions

- `train_test_sampling()`: Split your dataset into training and test
  sets.
- `lr()`: Train a logistic regression model.
- `predict_test()`: Make predictions on the test dataset.
- `predict_new()`: Make predictions on new data using a model trained
  with `lr()`.

## Conclusion

The `TrainPredict` package makes logistic regression modeling easy and
efficient by providing key functionalities for data splitting, model
training, and prediction. Whether you are a beginner in machine learning
or an experienced data scientist, `TrainPredict` can help you simplify
the modeling workflow.

## Get Involved

If you’d like to contribute or report any issues, please visit the
GitHub repository:

[TrainPredict on
GitHub](https://github.com/AU-R-Programming/Final_Project_Group_7)
