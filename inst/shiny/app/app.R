library(shiny)
library(dplyr)
library(TrainPredict)
# UI
ui <- fluidPage(
  titlePanel("TrainPredict Workflow: Train-Test Split, Logistic Regression, Predictions"),

  sidebarLayout(
    sidebarPanel(
      # Step 1: Train/Test Sampling
      h4("Step 1: Train/Test Split"),
      radioButtons("data_choice", "Choose Dataset",
                   choices = c("Upload CSV File", "mtcars", "iris"),
                   selected = "mtcars"),
      conditionalPanel(
        condition = "input.data_choice == 'Upload CSV File'",
        fileInput("data_file", "Upload CSV File", accept = ".csv")
      ),
      selectInput("dependent_var", "Dependent Variable", choices = NULL),
      numericInput("train_ratio", "Training Ratio (0-1)", value = 0.75),
      numericInput("seed", "Set Seed (Optional)", value = 123),
      checkboxInput("return_data", "Return Train/Test Datasets", value = TRUE),
      actionButton("split_button", "Split Data"),
      hr(),

      # Step 2: Logistic Regression
      h4("Step 2: Logistic Regression"),
      selectInput("response_var", "Response Variable", choices = NULL),
      selectInput("predictor_vars", "Predictor Variables", choices = NULL, multiple = TRUE),
      numericInput("B", "Number of Bootstrap Samples (B)", value = 20),
      numericInput("alpha", "Significance Level (alpha)", value = 0.05, min = 0, max = 1),
      actionButton("lr_button", "Run Logistic Regression"),
      hr(),

      # Step 3: Predict on Test Data
      h4("Step 3: Predict on Test Data"),
      actionButton("predict_test_button", "Predict on Test Data"),
      hr(),

      # Step 4: Generate New Data and Predict
      h4("Step 4: Generate New Data and Predict"),
      numericInput("new_obs", "Number of New Observations to Generate", value = 10, min = 1),
      numericInput("threshold", "Prediction Threshold", value = 0.5),
      actionButton("predict_new_button", "Generate New Data and Predict")
    ),

    mainPanel(
      h3("Results"),
      verbatimTextOutput("results"),
      tableOutput("output_table")
    )
  )
)

# Server
server <- function(input, output, session) {

  # Reactive dataset based on user input
  dataset <- reactive({
    if (input$data_choice == "Upload CSV File") {
      req(input$data_file)
      read.csv(input$data_file$datapath)
    } else if (input$data_choice == "mtcars") {
      mtcars
    } else if (input$data_choice == "iris") {
      iris %>% mutate(Virginica = Species == "virginica")
    }
  })

  # Reactive values to store results
  rv <- reactiveValues(
    train_data = NULL,
    test_data = NULL,
    model = NULL,
    model_formula = NULL,
    dependent_var_name = NULL,
    new_data = NULL
  )

  # Update variable choices dynamically based on dataset
  observe({
    req(dataset())
    column_names <- colnames(dataset())
    updateSelectInput(session, "dependent_var", choices = column_names)
    updateSelectInput(session, "response_var", choices = column_names)
    updateSelectInput(session, "predictor_vars", choices = column_names)
  })

  # Step 1: Train/Test Split
  observeEvent(input$data_choice, {
    req(dataset())

    # Update dependent variable choices based on the selected dataset
    if (input$data_choice == "mtcars") {
      updateSelectInput(session, "dependent_var", choices = c("am", "vs"))
    } else if (input$data_choice == "iris") {
      updateSelectInput(session, "dependent_var", choices = "Virginica")
    } else if (input$data_choice == "Upload CSV File") {
      # Allow selection from all columns if the user uploads a CSV file
      column_names <- colnames(dataset())
      updateSelectInput(session, "dependent_var", choices = column_names)
    }
  })

  # Step 1: Train/Test Split
  observeEvent(input$split_button, {
    req(dataset(), input$dependent_var, input$train_ratio)

    # Set the seed if provided
    seed_value <- ifelse(is.na(input$seed), NULL, input$seed)

    # Split data
    tryCatch({
      split_data <- train_test_sampling(
        data = dataset(),
        dependent_var = input$dependent_var,
        train_prop = input$train_ratio,
        return_data = input$return_data,
        seed = seed_value
      )

      if (input$return_data) {
        rv$train_data <- split_data$train
        rv$test_data <- split_data$test
        rv$dependent_var_name <- input$dependent_var

        # Output results: Head of train and test datasets, dimensions of complete, train, and test datasets
        output$results <- renderPrint({
          cat("Dimensions of Complete Dataset:\n")
          print(dim(dataset()))
          cat("\nDimensions of Training Dataset:\n")
          print(dim(rv$train_data))
          cat("\nDimensions of Testing Dataset:\n")
          print(dim(rv$test_data))
        })

        # Output table: Show head of train and test datasets
        output$output_table <- renderTable({
          head_data <- list(
            "Training Data (Head)" = head(rv$train_data),
            "Testing Data (Head)" = head(rv$test_data)
          )
          do.call(rbind, head_data)
        }, rownames = TRUE)

      } else {
        output$results <- renderPrint("Train/Test split indices generated successfully.")
        rv$train_data <- NULL
        rv$test_data <- NULL
        rv$dependent_var_name <- NULL
        output$output_table <- renderTable(NULL)
      }

    }, error = function(e) {
      output$results <- renderPrint(paste("Error:", e$message))
    })
  })


  # Step 2: Logistic Regression
  # Update response variable choices based on selected dataset for logistic regression
  observeEvent(input$data_choice, {
    req(dataset())

    # Update dependent variable choices based on the selected dataset
    if (input$data_choice == "mtcars") {
      updateSelectInput(session, "response_var", choices = c("am", "vs"))
    } else if (input$data_choice == "iris") {
      updateSelectInput(session, "response_var", choices = "Virginica")
    } else if (input$data_choice == "Upload CSV File") {
      # Allow selection from all columns if the user uploads a CSV file
      column_names <- colnames(dataset())
      updateSelectInput(session, "response_var", choices = column_names)
    }
  })

  # Step 2: Logistic Regression
  observeEvent(input$lr_button, {
    req(rv$train_data, input$response_var, input$predictor_vars, input$B, input$alpha)

    # Validate that the response variable selection is valid based on the dataset
    if (input$data_choice == "mtcars" && !(input$response_var %in% c("am", "vs"))) {
      output$results <- renderPrint("Error: For 'mtcars', only 'am' or 'vs' can be selected as the dependent variable.")
      return()
    }
    if (input$data_choice == "iris" && input$response_var != "Virginica") {
      output$results <- renderPrint("Error: For 'iris', only 'Virginica' can be selected as the dependent variable.")
      return()
    }

    # Run logistic regression
    tryCatch({
      formula <- as.formula(paste(input$response_var, "~", paste(input$predictor_vars, collapse = "+")))
      rv$model <- lr(formula = formula, data = rv$train_data, B = input$B, alpha = input$alpha)
      rv$model_formula <- formula

      # Output results: Detailed components from the logistic regression model
      output$results <- renderPrint({
        cat("Logistic Regression Results:\n\n")

        cat("Optimized Beta Coefficients (beta_optimized):\n")
        print(rv$model$beta_optimized)

        cat("\nConfidence Intervals (CI):\n")
        print(rv$model$CI)

        cat("\nConfusion Matrix:\n")
        print(rv$model$confusion_matrix)

        cat("\nPrevalence:\n")
        print(rv$model$prevalence)

        cat("\nAccuracy:\n")
        print(rv$model$accuracy)

        cat("\nSensitivity:\n")
        print(rv$model$sensitivity)

        cat("\nSpecificity:\n")
        print(rv$model$specificity)

        cat("\nFalse Discovery Rate:\n")
        print(rv$model$false_discovery_rate)

        cat("\nDiagnostic Odds Ratio:\n")
        print(rv$model$diagnostic_odds_ratio)
      })

      output$output_table <- renderTable(NULL) # Clear the table for this step
    }, error = function(e) {
      output$results <- renderPrint(paste("Error:", e$message))
    })
  })


  # Step 3: Predict on Test Data
  observeEvent(input$predict_test_button, {
    req(rv$test_data, rv$model, rv$dependent_var_name, rv$model_formula)

    # Predict on test data
    tryCatch({
      # Get prediction results from the predict_test function
      predictions <- predict_test(new_data = rv$test_data, model = rv$model,
                                  dependent_variable_col = rv$dependent_var_name)

      # Extract actual outcomes and predicted outcomes from the predictions
      actual_outcomes <- as.numeric(as.factor(rv$test_data[[rv$dependent_var_name]])) - 1
      predicted_outcomes <- predictions$predicted_outcomes

      # Ensure both actual and predicted have the same length
      if (length(actual_outcomes) != length(predicted_outcomes)) {
        stop("Mismatch in length between actual and predicted outcomes.")
      }

      # Create a confusion matrix
      confusion_matrix <- table(Actual = actual_outcomes, Predicted = predicted_outcomes)

      # Output results: Include model formula, dependent variable name, and the confusion matrix
      output$results <- renderPrint({
        cat("Using Model for Prediction:\n")
        cat("Model Formula:", deparse(rv$model_formula), "\n")
        cat("Dependent Variable:", rv$dependent_var_name, "\n")
        cat("Prediction on Test Data Completed.\n\n")

        cat("Confusion Matrix:\n")
        print(confusion_matrix)
      })

      # Show prediction results in the output table (excluding confusion matrix)
      output$output_table <- renderTable(predictions)

    }, error = function(e) {
      output$results <- renderPrint(paste("Error:", e$message))
    })
  })

  # Step 4: Generate New Data and Predict
  observeEvent(input$predict_new_button, {
    req(dataset(), rv$model, input$new_obs, input$threshold, input$predictor_vars)

    # Generate new random data for the predictor variables only
    predictor_vars <- input$predictor_vars
    num_obs <- input$new_obs

    # Generate random values ensuring consistent length across all predictor columns
    new_data_generated <- as.data.frame(lapply(dataset()[, predictor_vars, drop = FALSE], function(column) {
      if (is.numeric(column)) {
        # Generate random numeric values between min and max for numeric columns
        runif(num_obs, min = min(column, na.rm = TRUE), max = max(column, na.rm = TRUE))
      } else if (is.factor(column) || is.character(column)) {
        # Remove NA values from unique levels
        unique_levels <- na.omit(unique(column))
        if (length(unique_levels) == 0) {
          rep(NA, num_obs) # If no levels are left, fill with NA
        } else {
          # Sample from the unique levels for categorical columns
          sample(unique_levels, num_obs, replace = TRUE)
        }
      } else {
        # In case of other types (e.g., logical), we replicate a default value
        rep(column[1], num_obs)
      }
    }))

    # Ensure the new generated data has consistent number of rows
    new_data_generated <- as.data.frame(new_data_generated)

    if (nrow(new_data_generated) != num_obs) {
      output$results <- renderPrint("Error: Generated new data has inconsistent row count.")
      return()
    }

    # Store the new data in reactive values
    rv$new_data <- new_data_generated

    # Predict on new generated data
    tryCatch({
      predictions <- predict_new(data = rv$new_data, model = rv$model, threshold = input$threshold)


      # Combine predictor variables and predicted values into a new data frame
      prediction_results <- cbind(new_data_generated, Predicted_Value = predictions$predicted)

      # Output results
      output$results <- renderPrint("Prediction on new generated data completed.")
      output$output_table <- renderTable(prediction_results)
    }, error = function(e) {
      output$results <- renderPrint(paste("Error:", e$message))
    })
  })


}

# Run the app
shinyApp(ui = ui, server = server)
