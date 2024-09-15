# Load necessary libraries
library(dplyr)
library(ggplot2)
library(writexl)

# Directory to save results
data_dir <- "C:/Users/Usuario/Downloads/TFM/Code/data/raw"

# Number of samples
n_samples <- 800

# Probability vector for race
prob <- c(0.1, 0.9)  # Probability vector: 10% for 0 (minority), 90% for 1 (majority)

# Coefficients (hypothetical)
beta_0 <- 0.5
beta_1 <- -0.1
beta_2 <- -0.05
beta_3 <- 0.1
beta_4 <- 0.2  # Ccoefficient for minority status to simulate potential bias

# Loop to generate and save data and histograms 50 times
for (i in 1:10) {
  
  # Generate independent variables
  priors <- rnorm(n_samples, 2, 1)  # Number of Prior Years Incarcerated
  rehab <- sample(0:1, n_samples, replace = TRUE)  # Participation in Rehabilitation Programs (binary)
  
  # Generate sensitive variable (race) with more 1s than 0s
  S <- sample(0:1, n_samples, replace = TRUE, prob = prob)
  
  # Generate dependent variable (Sentence) based on S (race)
  sentence <- ifelse(S == 1, rnorm(n_samples, mean = 4, sd = 1.2), rnorm(n_samples, mean = 6, sd = 1.2))
  
  # Generate dependent variable (Likelihood of Parole) based on S
  Y <- beta_0 + beta_1 * priors + beta_2 * sentence + beta_3 * rehab + beta_4 * S + error
  
  # Normalize function
  normalize <- function(x) {
    (x - min(x)) / (max(x) - min(x))
  }
  
  # Normalize the Y values
  Y <- normalize(Y)  # Ensure likelihood is between 0 and 1
  
  # Convert to binary
  Y <- ifelse(Y > 0.5, 1, 0)
  
  # Create a data frame
  data <- data.frame(priors = priors, sentence = sentence, rehab = rehab, S = S, Y = Y)
  
  # Save the data frame as an Excel file
  file_name <- paste0(data_dir, "/datacorr", i, ".xlsx")
  write_xlsx(data, path = file_name)
  
  # Fit the linear regression model
  model <- lm(Y ~ priors + sentence + rehab + S, data = data)
  
  # Predict on the training data
  data$predicted_Y <- predict(model, data)
  
  # Evaluate fairness: Compare mean predictions for different groups
  mean_pred_group_0 <- mean(data$predicted_Y[data$S == 0])
  mean_pred_group_1 <- mean(data$predicted_Y[data$S == 1])
  
  cat("Iteration", i, ":\n")
  cat("Mean prediction for group 0 (Minority):", mean_pred_group_0, "\n")
  cat("Mean prediction for group 1 (Majority):", mean_pred_group_1, "\n")
  
  # Create and save the histogram of predicted values
  hist_plot <- ggplot(data, aes(x = factor(S), y = predicted_Y)) +
    geom_boxplot() +
    labs(title = paste("Predicted Likelihood of Parole by Race (Iteration", i, ")"),
         x = "Race (0 = Minority, 1 = Majority)",
         y = "Predicted Likelihood of Parole") +
    theme_minimal()
  
  # Save the histogram plot
  hist_file_name <- paste0(data_dir, "/histcorr", i, ".png")
  ggsave(filename = hist_file_name, plot = hist_plot)
}




### Key Changes:
# 1. **Correlation in Data Generation**: The `generate_correlated_data` function is used to generate `priors` 
# and `sentence` with a specified correlation (rho = 0.3 in this case).
#2. **Correlation between `priors` and `sentence`**: This simulates a realistic scenario where individuals with
# more prior incarcerations might also receive longer sentences.

#This code will generate 50 datasets, each with correlated variables, and save the data and histograms as 
# specified. Adjust the correlation value (`rho`) to better fit your specific context if needed.