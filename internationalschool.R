# IX International School 2025
# A decision tree approach for sugarcane area classification
# Clarice Demétrio, Cristian Villegas e Marcelo da Silva
# Department of Exact Sciences
# ''Luiz de Queiroz'' College of Agriculture
# University of São Paulo, Piracicaba, Brazil

# Download and Install R
# https://cran.r-project.org

# Download and Install RStudio
# https://posit.co/download/rstudio-desktop


# Remove everything in the working environment
rm(list=ls())

# Reads a .csv file and creates a data frame from it
# Observations corresponding to rows and variables to columns
sim_df <- read.csv("simulateddata.csv", header = TRUE, sep = ";", dec = ",")

# View data in a spreadsheet and another tab
View(sim_df)

# Structure of the data
str(sim_df)

# Summary of the data
summary(sim_df)

# A little more detail about the variable called Groups
table(sim_df$Groups)

# Transform Groups in factor (category variable)
sim_df$Groups <- as.factor(sim_df$Groups)


# --- Visualization of data distribution ---

# Set up a 1-row, 5-column plot layout
par(mfrow = c(1, 5)) 

# Create boxplots to compare the distribution of variables across different Groups
boxplot(X1 ~ Groups, data = sim_df, main = "Distribution of \n X1 by Groups",
        xlab = "Groups", ylab = "X1", col = c("#00A9FF", "#FF61CC"))
boxplot(X2 ~ Groups, data = sim_df, main = "Distribution of \n X2 by Groups",
        xlab = "Groups", ylab = "X2", col = c("#00A9FF", "#FF61CC"))
boxplot(X3 ~ Groups, data = sim_df, main = "Distribution of \n X3 by Groups",
        xlab = "Groups", ylab = "X3", col = c("#00A9FF", "#FF61CC"))
boxplot(X4 ~ Groups, data = sim_df, main = "Distribution of \n X4 by Groups",
        xlab = "Groups", ylab = "X4", col = c("#00A9FF", "#FF61CC"))
boxplot(X5 ~ Groups, data = sim_df, main = "Distribution of \n X5 by Groups",
        xlab = "Groups", ylab = "X5", col = c("#00A9FF", "#FF61CC"))

# Set up a 1-row, 1-column plot layout (reset)
par(mfrow = c(1, 1)) 

# --- Data Splitting Code ---

# Install the 'caret' package if you haven't already
# install.packages("caret")

# Load the 'caret' package
library(caret)

# Set the desired proportion for the training set (e.g., 70% for training)
train_proportion <- 0.7

# Create an index for the training data using stratified sampling
set.seed(456)
train_index <- createDataPartition(sim_df$Groups, 
                                   p = train_proportion, 
                                   list = FALSE)

# Split the data into training and testing sets
train_data <- sim_df[train_index, ]
test_data <- sim_df[-train_index, ]

# --- Classification tree method ---

# Install the 'rpart' package if you haven't already
# install.packages("rpart")

# Load the 'rpart' package
library(rpart)

# Train the classification tree
tree_model <- rpart(Groups ~ ., data = train_data, method = "class")

# Install the 'rpart.plot' package if you haven't already
# install.packages("rpart.plot")

# Load the 'rpart.plot' package
library(rpart.plot)

# Plot the tree for visualization
rpart.plot(tree_model, extra = 101, fallen.leaves = TRUE, type = 3, cex = 0.8)

# --- Evaluation on the Test Set (classification tree) ---

# Make predictions on the test data
predictions_class_tree <- predict(tree_model, newdata = test_data, type = "class")
predictions_prob_tree <- predict(tree_model, newdata = test_data, type = "prob")

# Generate the Confusion Matrix
cm_tree <- confusionMatrix(predictions_class_tree, 
                           test_data$Groups, 
                           positive = "2")

# Print the confusion matrix and related statistics
print(cm_tree)

# Extract individual metrics for clarity
cat("\n--- Classification Metrics ---\n")
cat("Accuracy:", cm_tree$overall['Accuracy'], "\n")
cat("Sensitivity (True Positive Rate):", cm_tree$byClass['Sensitivity'], "\n")
cat("Specificity (True Negative Rate):", cm_tree$byClass['Specificity'], "\n")

# Install the 'pROC' package if you haven't already
# install.packages("pROC")

# Load the 'pROC' package
library(pROC)

# Build a ROC curve
roc_curve_tree <- roc(response = test_data$Groups,
                      predictor = predictions_prob_tree[, 2], 
                      levels = levels(test_data$Groups))

# Plot the ROC curve
plot(roc_curve_tree, legacy.axes = TRUE, print.auc = TRUE,
     main = "ROC Curve for Classification Tree",
     col = "#1c61b6", lwd = 2)

# --- Random forest method ---

# Install the 'randomForest' package if you haven't already
# install.packages("randomForest")

# Load the 'randomForest' package
library(randomForest)

# For reproducibility of the Random Forest model training
set.seed(789)

# Train the Random Forest
rf_model <- randomForest(Groups ~ .,
                         data = train_data,
                         ntree = 500,
                         mtry = floor(sqrt(ncol(train_data) - 1)),
                         importance = TRUE)

# --- Evaluation on the Test Set (Random Forest) ---

# Make predictions on the test data
predictions_class_rf <- predict(rf_model, newdata = test_data, type = "class")
predictions_prob_rf <- predict(rf_model, newdata = test_data, type = "prob")

# Generate the Confusion Matrix
cm_rf <- confusionMatrix(predictions_class_rf, 
                         test_data$Groups, 
                         positive = "2")

# Print the confusion matrix and related statistics
print(cm_rf)

# Extract individual metrics for clarity
cat("\n--- Classification Metrics ---\n")
cat("Accuracy:", cm_rf$overall['Accuracy'], "\n")
cat("Sensitivity (True Positive Rate):", cm_rf$byClass['Sensitivity'], "\n")
cat("Specificity (True Negative Rate):", cm_rf$byClass['Specificity'], "\n")

# Build a ROC curve
roc_curve_rf <- roc(response = test_data$Groups,
                    predictor = predictions_prob_rf[, 2], 
                    levels = levels(test_data$Groups))

# Plot the ROC curve
plot(roc_curve_rf, legacy.axes = TRUE, print.auc = TRUE,
     main = "ROC Curve for Classification Tree",
     col = "#1c61b6", lwd = 2)

