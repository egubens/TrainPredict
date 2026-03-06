#' @title Train-Test Sampling Function
#'
#' @description This function splits a data set into training and test sets based on a specified proportion of training data, while ensuring that the split maintains the class distribution of a binomial dependent variable.
#'
#' @param data A data frame containing the data to be split. The data should have no missing values in the rows intended for modeling.
#' @param dependent_var A character string or numeric value indicating the name or column index of the dependent (target) variable in `data`. The dependent variable must be binomial (i.e., it should have exactly two unique levels).
#' @param train_prop A numeric value between 0 and 1 indicating the proportion of the data to include in the training set. The default value is 0.75.
#' @param return_data A logical value indicating whether to return the actual training and test datasets (`TRUE`) or just the training indices (`FALSE`). Default is `FALSE`.
#' @param seed An optional integer to set the random seed for reproducibility.
#'
#' @details The function first removes any incomplete cases from the data and then checks whether the dependent variable is binomial. It then proceeds to split the data into training and test sets based on the specified proportion (`train_prop`). If `return_data = TRUE`, the function returns a list containing the training and test datasets. Otherwise, it returns only the indices of the rows assigned to the training set.
#'
#' @return Depending on the value of `return_data`, the function returns:
#' 	- If `return_data = FALSE`: A vector of indices for the rows in the training set.
#' 	- If `return_data = TRUE`: A list with two elements:
#' 	  	- `train`: A data frame containing the training set.
#' 	  	- `test`: A data frame containing the test set.
#'
#' @examples
#' # Example 1: Using the mtcars dataset with 'am' as the binomial dependent variable.
#' # Split the data, using 75% of the data for training, and return the train and test datasets.
#'
#' split_data <- train_test_sampling(mtcars, dependent_var = 'am', train_prop = 0.75, return_data = TRUE, seed = 123)
#'
#' #To obtain the training data set can be found in an element called \code{train}
#' head(split_data$train)
#' dim(split_data$train)
#'
#' #The test data set is stored in an element called \code{test}
#' head(split_data$test)
#' dim(split_data$test)
#'
#' # Get the training indices only
#' train_indices <- train_test_sampling(mtcars, dependent_var = 'am', train_prop = 0.75, return_data = FALSE, seed = 123)
#'
#' # Example 2: Splitting the Iris dataset (binary classification case)
#' data(iris)
#' iris_binary <- iris[iris$Species %in% c("setosa", "versicolor"), ]
#' iris_binary$Species <- factor(iris_binary$Species)
#'
#' # Get indices for the training set
#' train_indices <- train_test_sampling(data = iris_binary, dependent_var = "Species", train_prop = 0.8)
#' head(train_indices)
#'
#' # Split data into training and testing sets
#' split_data <- train_test_sampling(data = iris_binary, dependent_var = "Species", train_prop = 0.8, return_data = TRUE)
#' head(split_data$train)
#' head(split_data$test)
#'
#' @seealso \code{\link[TrainPredict]{lr}}, \code{\link[TrainPredict]{predict_test}}
#'
#' @export
train_test_sampling <- function(data, dependent_var, train_prop = 0.75, return_data = FALSE, seed = NULL) {
  # Ensure the data is a dataframe
  data <- as.data.frame(data)

  # Remove incomplete cases
  data <- data[complete.cases(data), ]

  # Handle dependent_var as either column index or name
  if (is.numeric(dependent_var)) {
    if (dependent_var > ncol(data) || dependent_var < 1) {
      stop("The specified dependent variable index is out of range.")
    }
    dependent_var <- names(data)[dependent_var]
  } else if (!(dependent_var %in% names(data))) {
    stop("The specified dependent variable is not in the data.")
  }

  # Check if the dependent variable is binomial
  unique_levels <- unique(data[[dependent_var]])
  if (length(unique_levels) != 2) {
    stop("The dependent variable must be binomial with exactly two unique levels.")
  }

  # Set seed if provided
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Split the data by levels of the dependent variable and collect training indices
  train_indices <- unlist(lapply(unique_levels, function(level) {
    subset_data <- which(data[[dependent_var]] == level)
    sample(subset_data, size = round(length(subset_data) * train_prop), replace = FALSE)
  }))

  # Return the training indices only or the train and test datasets
  if (return_data) {
    train_data <- data[train_indices, ]
    test_data <- data[-train_indices, ]
    return(list(train = train_data, test = test_data))
  } else {
    return(train_indices)
  }
}

#' @title Logistic Regression with Bootstrapping and Evaluation Metrics
#'
#' @description This function fits a logistic regression model using either a formula and data frame or
#' user-provided independent and dependent variables (X and y). It also computes bootstrap
#' confidence intervals, evaluation metrics such as accuracy, sensitivity, specificity, and
#' diagnostic odds ratio.
#'
#' @param formula Optional. A formula object specifying the response and predictor variables.
#' @param data Optional. A data frame containing the data referenced in the formula.
#' @param X Optional. A matrix or data frame of predictor variables.
#' @param y Optional. A vector of binary response values (0 or 1).
#' @param B Integer. The number of bootstrap samples to generate. Default is 20.
#' @param alpha Numeric. The significance level for confidence interval estimation. Default is 0.05.
#'
#' @return A list containing:
#' \item{beta_init}{The initial coefficients estimated using the least squares approach.}
#' \item{beta_optimized}{The optimized coefficients obtained using maximum likelihood estimation.}
#' \item{CI}{A matrix containing the bootstrap confidence intervals for each coefficient.}
#' \item{confusion_matrix}{A 2x2 confusion matrix comparing actual vs. predicted values.}
#' \item{prevalence}{The prevalence (proportion of positive cases) in the response variable.}
#' \item{accuracy}{The accuracy of the model, defined as the proportion of correct predictions.}
#' \item{sensitivity}{The sensitivity (true positive rate) of the model.}
#' \item{specificity}{The specificity (true negative rate) of the model.}
#' \item{false_discovery_rate}{The false discovery rate of the model.}
#' \item{diagnostic_odds_ratio}{The diagnostic odds ratio, defined as (TP / FN) / (FP / TN).}
#' \item{factor_mappings}{A list of data frames mapping factor levels to numeric values for all categorical variables.}
#'
#' @details
#' The \code{lr} function performs logistic regression with an option to input the formula and data directly or
#' provide independent variables (X) and response variable (y) separately. If categorical variables are present
#' in the data, they are converted to dummy variables.
#'
#' The function computes the logistic regression model using maximum likelihood estimation. It also generates
#' bootstrap samples to estimate confidence intervals for the model coefficients. Performance metrics, including
#' accuracy, sensitivity, specificity, false discovery rate, and diagnostic odds ratio, are calculated based on
#' the predictions.
#'
#' @examples
#' # Example 1: using the mtcars dataset with 'am' as the binomial dependent variable.
#' # Split the data, using 75% of the data for training, and return the train and test data sets.
#' data(mtcars)
#' data <- train_test_sampling(mtcars, "am", train_prop=0.75, return_data=TRUE, seed=123)
#' train<-data$train
#' result <- lr(formula = am ~ mpg + hp + wt, data = train)
#' print(result$beta_optimized)
#' print(result$confusion_matrix)
#'
#' # Example 2: Using data set iris
#' data(iris)
#' # Create a binary dependent variable that includes only species 'setosa' and 'versicolor'
#' iris_binary <- iris[iris$Species %in% c("setosa", "versicolor"), ]
#' iris_binary$Species <- factor(iris_binary$Species)
#'
#' # Fit the model
#' result <- lr(Species ~ Sepal.Length + Sepal.Width, data = iris_binary)
#' print(result$beta_optimized)
#'
#'# Example 3: Using X and y directly
#' X <- iris_binary[, c("Sepal.Length", "Sepal.Width")]
#' y <- as.numeric(iris_binary$Species) - 1
#' result <- lr(X = X, y = y, B = 50, alpha = 0.01)
#' print(result$beta_optimized)
#'
#' @note
#' The response variable (\code{y}) must have exactly two levels for logistic regression to be applicable.
#' All categorical variables are internally converted to numeric variables.
#'
#' @seealso \code{\link[TrainPredict]{predict_new}}, \code{\link[TrainPredict]{predict_test}}, \code{\link[TrainPredict]{train_test_sampling}}
#'
#' @export
lr <- function(formula = NULL, data = NULL, X = NULL, y = NULL, B = 20, alpha = 0.05) {

  # Store factor level mapping information
  factor_mappings <- list()

  # Determine if the user provided a formula or X and y directly
  if (!is.null(formula) && !is.null(data)) {
    # Store factors
    for (col_name in names(data)) {
      col <- data[[col_name]]
      if (is.character(col) || is.factor(col) || is.logical(col)) {
        levels_info <- levels(as.factor(col))
        factor_mappings[[col_name]] <- data.frame(
          Level = levels_info,
          Numeric = as.numeric(as.factor(levels_info)) - 1
        )
        data[[col_name]] <- as.numeric(as.factor(col)) - 1
      }
    }

    # Parse formula to extract response and predictor variables
    mf <- model.frame(formula, data)
    y <- model.response(mf)
    X <- model.matrix(formula, data)

    # Remove the intercept from the model matrix since we'll add it manually
    X <- X[, -1, drop = FALSE]
  } else if (!is.null(X) && !is.null(y)) {
    # Make sure X is a data frame for easier column-wise conversion
    if (!is.data.frame(X)) {
      X <- as.data.frame(X)
    }

    # Convert character or factor covariates to dummy variables
    for (col_name in names(X)) {
      col <- X[[col_name]]
      if (is.character(col) || is.factor(col) || is.logical(col)) {
        levels_info <- levels(as.factor(col))
        factor_mappings[[col_name]] <- data.frame(
          Level = levels_info,
          Numeric = as.numeric(as.factor(levels_info)) - 1
        )
        X[[col_name]] <- as.numeric(as.factor(col)) - 1
      }
    }

    # Convert character or factor covariates to dummy variables
    X <- model.matrix(~ . - 1, data = X)
  } else {
    stop("Please provide either a formula and data, or X and y.")
  }

  # Store the names of the coefficients for later use
  coef_names <- colnames(X)

  # Remove the suffix TRUE/FALSE from coefficient names if present
  coef_names <- gsub("TRUE|FALSE", "", coef_names)

  # Convert dependent variable to binary if it is in character or factor form
  if (is.character(y) || is.factor(y) || is.logical(y)) {
    unique_levels <- unique(y)
    if (length(unique_levels) != 2) {
      stop("The dependent variable must have exactly two levels to be used in logistic regression.")
    }
    # Convert character or factor levels to 0 and 1
    y <- as.numeric(y == unique_levels[2])
  } else {
    # Convert numeric dependent variable to 0 and 1 if it has more than two unique values
    unique_levels <- unique(y)
    if (length(unique_levels) != 2) {
      stop("The dependent variable must have exactly two levels to be used in logistic regression.")
    }
    y <- as.numeric(y == max(unique_levels))
  }

  # Ensure y values are binary (0 or 1)
  if (any(y < 0 | y > 1)) {
    stop("The dependent variable must have values between 0 and 1.")
  }

  # Create design matrix with intercept
  design <- cbind(Intercept = rep(1, nrow(X)), X)

  # Update coef_names to include the intercept
  coef_names <- c("Intercept", coef_names)

  # Initialize beta with least squares formula
  beta_init <- solve(t(design) %*% design) %*% t(design) %*% y

  # Define the negative log-likelihood
  neg_log_likelihood <- function(beta) {
    p <- 1 / (1 + exp(-design %*% beta))
    -sum(y * log(p) + (1 - y) * log(1 - p))
  }

  # Optimization using optim
  result <- optim(beta_init, neg_log_likelihood)

  # Create bootstrap data matrix with y and X
  booth_data <- cbind(y, design)

  n <- nrow(booth_data)

  # Matrices to store bootstrap coefficients
  B_hat <- B_hat2 <- matrix(NA, nrow = B, ncol = ncol(booth_data) - 1)

  # Loop to create bootstrap samples
  for (i in 1:B) {
    # Sampling the data for bootstrap
    bdata <- as.matrix(booth_data[sample(1:n, n, replace = TRUE), ])

    # Separating covariates
    Xs <- bdata[, -1]

    # Separating dependent variable
    ys <- bdata[, 1]

    # Calculating coefficients of sampled data
    beta_init2 <- solve(t(Xs) %*% Xs) %*% t(Xs) %*% ys

    # Calculating coefficients with optimization
    boot_lm <- optim(beta_init2, neg_log_likelihood)

    # Calculating coefficients with glm
    boot_lm2 <- suppressWarnings(glm(ys ~ Xs[, -1], family = binomial))

    # Storing bootstrap coefficients
    B_hat[i, ] <- boot_lm$par

    # Storing glm coefficients
    B_hat2[i, ] <- boot_lm2$coefficients
  }

  # Confidence interval based on bootstrap
  CI <- matrix(NA, nrow = ncol(booth_data) - 1, ncol = 2)

  # Loop to calculate CI for coefficients obtained with bootstrap
  for (i in 1:ncol(B_hat)) {
    CI[i, ] <- quantile(B_hat[, i], c(alpha / 2, 1 - alpha / 2))
  }

  # Set row names of CI to coefficient names
  rownames(CI) <- coef_names

  # Predict probabilities based on optimized beta coefficients
  p_hat <- 1 / (1 + exp(-design %*% result$par))

  # Set threshold for classification
  y_pred <- ifelse(p_hat >= 0.5, 1, 0)

  # Create confusion matrix
  true_positive <- sum(y == 1 & y_pred == 1)
  true_negative <- sum(y == 0 & y_pred == 0)
  false_positive <- sum(y == 0 & y_pred == 1)
  false_negative <- sum(y == 1 & y_pred == 0)

  confusion_matrix <- matrix(c(false_positive, true_positive, true_negative, false_negative),
                             nrow = 2,
                             dimnames = list(
                               "Actual" = c("FALSE", "TRUE"),
                               "Predicted" = c("TRUE", "FALSE")
                             ))

  # Calculate accuracy, sensitivity, specificity, false discovery rate, and diagnostic odds ratio
  prevalence <- mean(y)
  accuracy <- (true_positive + true_negative) / length(y)
  sensitivity <- true_positive / (true_positive + false_negative)
  specificity <- true_negative / (true_negative + false_positive)
  false_discovery_rate <- false_positive / (true_positive + false_positive)
  diagnostic_odds_ratio <- (true_positive / false_negative) / (false_positive / true_negative)

  return(list(beta_init = beta_init,
              beta_optimized = result$par,
              CI = CI,
              confusion_matrix = confusion_matrix,
              prevalence = prevalence,
              accuracy = accuracy,
              sensitivity = sensitivity,
              specificity = specificity,
              false_discovery_rate = false_discovery_rate,
              diagnostic_odds_ratio = diagnostic_odds_ratio,
              factor_mappings = factor_mappings))
}

#' @title Predict Outcomes and Evaluate Model Performance on Test Data Set
#'
#' @description This function predicts outcomes for a new data set using a provided logistic regression model obtained with function \code{lr} and compares these predictions to the actual outcomes using a confusion matrix. The model must contain optimized beta coefficients (`beta_optimized`) and factor mappings (`factor_mappings`). The function allows for numeric conversion of factor variables in the new data.
#'
#' @param new_data A data frame containing the new observations to predict. The column names should match the predictor names in the model.
#' @param model A list containing the logistic regression model details. It must have two components:
#' 	- `beta_optimized`: A matrix of optimized beta coefficients, including an intercept term.
#' 	- `factor_mappings`: A list where each element represents the mapping of levels to numeric values for factor variables.
#' @param dependent_variable_col A character string specifying the name or number of the column in `new_data` containing the actual outcomes for evaluation purposes.
#'
#' @details This function first converts any factor variables in the new data to numeric values based on the mappings provided in the model. Then, it extracts the predictors required by the model, calculates the log-odds, and converts them to probabilities using the logistic function. Predictions are classified as either "TRUE" or "FALSE" based on a probability threshold of 0.5. A confusion matrix is printed to assess the accuracy of the predictions.
#'
#' @return A data frame with two columns:
#'  - `actual_outcomes`: Actual values of the dependent variable in the test data set.
#' 	- `predicted_outcomes`: Predicted outcomes as "TRUE" or "FALSE".
#'#'
#' @examples
#' # Example 1: Using the mtcars dataset with 'am' as the binomial dependent variable.
#' # First create a training and test sets from original data using function \code{train_test_sampling}
#' split_data <- train_test_sampling(mtcars, dependent_var = 'am', train_prop = 0.75, return_data = TRUE, seed = 123)
#' train_data<-split_data$train
#' test_data<-split_data$test
#'
#' # For this example we will create a model object using function \code{lr} using the training data, and variable am as the dependent variable.
#' lrcars<-lr(am~hp+mpg+wt, data=train_data, B=50, alpha=0.05)
#'
#' # Now that we have created a logistic regression model with the training data, we test the performance of the model in new unseen data stored in the test data set.
#' results<-predict_test(test_data, lrcars, "am")
#' head(results)
#'
#' # Example 2: Predicting outcomes on a binary subset of the Iris dataset
#' data(iris)
#' iris_binary <- iris[iris$Species %in% c("setosa", "versicolor"), ]
#' iris_binary$Species <- factor(iris_binary$Species)
#'
#' # Create a training and test sets from original data using function \code{train_test_sampling}
#' iris_binary_samples<-train_test_sampling(iris_binary, dependent_var="Species", train_prop=0.75, return_data=TRUE, seed=123)
#' iris_train<-iris_binary_samples$train
#' iris_test<-iris_binary_samples$test
#'
#' # Fit a logistic regression model using function \code{lr}
#' model <- lr(Species ~ Sepal.Length + Sepal.Width, data = iris_train)
#'
#' # Predict outcomes on test data
#' predictions <- predict_test(new_data = iris_test, model = model, dependent_variable_col = "Species")
#' head(predictions)
#'
#' @seealso \code{\link[TrainPredict]{lr}}, \code{\link[TrainPredict]{train_test_sample}}, \code{\link[TrainPredict]{predict_new}}
#'
#' @export
predict_test <- function(new_data, model, dependent_variable_col) {
  # Extract beta_optimized and factor_mappings from the model
  beta_optimized <- model$beta_optimized
  factor_mappings <- model$factor_mappings

  # Convert factors to numeric based on factor mappings
  for (factor_name in names(factor_mappings)) {
    if (factor_name %in% colnames(new_data)) {
      mapping <- factor_mappings[[factor_name]]
      levels <- mapping$Level
      numeric_values <- mapping$Numeric
      new_data[[factor_name]] <- as.numeric(factor(new_data[[factor_name]], levels = levels, labels = numeric_values))
    }
  }

  # Extract the predictors from the model
  predictor_names <- rownames(beta_optimized)[-1]  # Exclude the intercept

  # Check if all predictors are present in the new_data
  missing_predictors <- setdiff(predictor_names, colnames(new_data))
  if (length(missing_predictors) > 0) {
    stop("The following predictors are missing in the new data: ", paste(missing_predictors, collapse = ", "))
  }

  # Ensure the order of columns in new_data matches the order in beta_optimized
  predictors_data <- new_data[, predictor_names, drop = FALSE]

  # Extract the predictors and add an intercept column
  X <- as.matrix(cbind(Intercept = 1, predictors_data))

  # Predict the log-odds
  log_odds <- X %*% beta_optimized

  # Convert log-odds to probabilities using the logistic function
  probabilities <- 1 / (1 + exp(-log_odds))

  # Classify as 'Positive' or 'Negative' based on a threshold of 0.5
  binary_outcomes <- ifelse(probabilities >= 0.5, "TRUE", "FALSE")

  # Add a column with binary outcomes (0 or 1)
  predicted_outcomes <- ifelse(binary_outcomes == "TRUE", 1, 0)

  # Compare the binary outcomes to the actual dependent variable
  actual_outcomes <- as.numeric(as.factor(new_data[[dependent_variable_col]]))-1

  # Create a data frame with both predicted outcomes and binary outcomes
  result <- data.frame(actual_outcomes, predicted_outcomes)

  # Convert actual outcomes to binary (0 or 1) for comparison if necessary
  if (is.factor(actual_outcomes) || is.character(actual_outcomes)) {
    actual_outcomes <- as.numeric(as.factor(actual_outcomes))-1
  }

  # Create a confusion matrix using binary outcomes
  confusion_matrix <- table(Predicted = predicted_outcomes, Actual = actual_outcomes)

  # Add the confusion matrix to the result
  print("Confusion Matrix:")
  print(confusion_matrix)

  return(result)
}

#' @title Generate Predictions for New Data Using a Logistic Regression Model
#'
#' @description This function generates predictions for new data based on an input logistic regression model fitted using function \code{lr}.
#' The model is should include a list that contains an optimized set of coefficients (betas).
#' Predictions are based on a logistic regression model using a specified threshold to classify outcomes.
#'
#' @param data A data frame containing the new data to make predictions on. The dataset should contain
#'   all predictors specified in the model.
#' @param model A list representing the logistic regression model. It must contain a component named
#'   \code{beta_optimized}, which is a column matrix containing the model's coefficients, including the intercept.
#' @param threshold A numeric value between 0 and 1. This value is used as the decision threshold for classifying
#'   outcomes as 0 or 1. The default value is 0.5.
#'
#' @return A data frame containing the original data along with an additional column named \code{predicted}, which
#'   holds the predicted class labels (0 or 1).
#'
#' @examples
#'# Example 1: Using the mtcars dataset with 'am' as the binomial dependent variable.
#'
#' # For this example we will create a model object using function \code{lr} using the training data, and variable 'am' as the dependent variable.
#' data(mtcars)
#' model<-lr(am~hp+mpg+wt, data=mtcars, B=50, alpha=0.05)
#'
#' # For this example we will generate random data for variables 'hp', 'mpg', and 'wt'.
#' n<-10
#' hp<-round(runif(n, min(mtcars$hp), max(mtcars$hp)),1)
#' mpg<-round(runif(n, min(mtcars$mpg), max(mtcars$mpg)),1)
#' wt<-round(runif(n, min(mtcars$wt), max(mtcars$wt)),1)
#'
#' new_data<-data.frame(
#'   hp=hp,
#'   wt=wt,
#'   mpg=mpg)
#'
#' #Now we use the function to predict the outcome
#' new_predictions<-predict_new(data=new_data, model=model, threshold=0.5)
#' head(new_predictions)
#'
#' # Example 2: Predicting outcomes on a binary subset of the Iris dataset
#' data(iris)
#' iris_binary <- iris[iris$Species %in% c("setosa", "versicolor"), ]
#' iris_binary$Species <- factor(iris_binary$Species)
#'
#' # Fit a logistic regression model using function \code{lr}
#' model <- lr(Species ~ Sepal.Length + Sepal.Width, data = iris_binary)
#'
#' # For this example we will generate new random data for variables 'Sepal.Length' and 'Sepal.Width'.
#' n<-10
#' Sepal.Length<-round(runif(n, min(iris_binary$Sepal.Length), max(iris_binary$Sepal.Length)), 1)
#' Sepal.Width<-round(runif(n, min(iris_binary$Sepal.Width), max(iris_binary$Sepal.Width)), 1)
#' new_data<-data.frame(
#'   Sepal.Length=Sepal.Length,
#'   Sepal.Width=Sepal.Width)
#'
#' # Predict outcomes on new data
#' predictions <- predict_new(data = new_data, model = model, threshold=0.5)
#' head(predictions)
#'
#' @seealso \code{\link[TrainPredict]{lr}}, \code{\link[TrainPredict]{train_test_sample}}, \code{\link[TrainPredict]{predict_test}}
#'
#' @export
predict_new <- function(data, model, threshold = 0.5) {
  # Ensure the data is a dataframe
  data1 <- as.data.frame(data)

  # Validate the model structure
  if (!is.list(model) || !"beta_optimized" %in% names(model) || !"factor_mappings" %in% names(model)) {
    stop("The model parameter must be a list containing elements 'beta_optimized' and 'factor_mappings'.")
  }
  beta_optimized <- model$beta_optimized
  factor_mappings <- model$factor_mappings

  # Validate beta_optimized
  if (!is.matrix(beta_optimized) || ncol(beta_optimized) != 1) {
    stop("The 'beta_optimized' parameter must be a column matrix.")
  }

  # Extract predictor names from beta_optimized, excluding the intercept
  predictor_names <- rownames(beta_optimized)[-1]  # Exclude the intercept

  # Check if all predictor names are present in the dataset
  missing_predictors <- setdiff(predictor_names, names(data1))
  if (length(missing_predictors) > 0) {
    stop(paste("The dataset is missing the following predictors:",
               paste(missing_predictors, collapse = ", ")))
  }

  # Convert categorical columns to numeric using factor_mappings
  for (factor_name in names(factor_mappings)) {
    if (factor_name %in% names(data1)) {
      mapping <- factor_mappings[[factor_name]]
      # Ensure the data column is a factor
      if (!is.factor(data1[[factor_name]])) {
        data1[[factor_name]] <- as.factor(data1[[factor_name]])
      }
      # Convert factor levels to numeric based on mapping
      levels(data1[[factor_name]]) <- mapping$Numeric[match(levels(data1[[factor_name]]), mapping$Level)]
      data1[[factor_name]] <- as.numeric(as.character(data1[[factor_name]]))
    }
  }

  # Select and preprocess necessary predictors
  data1 <- data1[, predictor_names, drop = FALSE]

  # Ensure all selected columns are numeric
  non_numeric_cols <- names(data1)[!sapply(data1, is.numeric)]
  if (length(non_numeric_cols) > 0) {
    stop(paste("The following columns must be numeric:",
               paste(non_numeric_cols, collapse = ", ")))
  }

  # Add intercept column to the data
  data_with_intercept <- cbind(Intercept = 1, data1)

  # Ensure the intercept and data matrix are numeric
  if (!is.numeric(as.matrix(data_with_intercept))) {
    stop("The data with intercept must be a numeric matrix.")
  }

  # Calculate the linear predictor
  linear_predictor <- as.matrix(data_with_intercept) %*% beta_optimized

  # Apply the logistic function to get probabilities
  probabilities <- 1 / (1 + exp(-linear_predictor))

  # Make predictions based on the threshold
  predictions <- ifelse(probabilities >= threshold, 1, 0)

  # Add the predictions as a new column
  data_with_predictions <- cbind(data, predicted = predictions)

  # Return the predictions and the updated dataset
  return(data_with_predictions)
}

