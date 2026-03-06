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
