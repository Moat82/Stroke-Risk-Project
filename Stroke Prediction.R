## Stroke Risk Prediction Using Statistical and Machine Learning Models


## Package and Data Set Input
library(dplyr)
library(janitor)
library(ggplot2)
library(tibble)
library(pROC)
library(caret)
# read healthcare_dataset_stroke_csv and assign it to the object 'sds'(stroke
# data set) for ease of use. Converts N/A to missing entries 
sds <- read.csv("healthcare_dataset_stroke_data.csv", header = TRUE,
                na.strings = c("N/A", ""))

# converts the response variable and the categorical predictors to factors so
# R can run them correctly
sds <- sds%>%
  mutate(
    stroke = factor(stroke, levels = c(0, 1), labels = c("No", "Yes")),            
    gender = factor(gender),
    hypertension = factor(hypertension),
    heart_disease = factor(heart_disease),
    ever_married = factor(ever_married),
    work_type = factor(work_type),
    Residence_type = factor(Residence_type),
    smoking_status = factor(smoking_status)
  )

# Data quality check to determine structure of dataset, also use table function 
# to determine if any errors in entries
names(sds)
dim(sds)
str(sds)
sum(is.na(sds))
colSums(is.na(sds))
table(sds$gender)
table(sds$ever_married)
table(sds$work_type)
table(sds$Residence_type)
table(sds$smoking_status)
str(sds)
# Gender has 1 entry as 'other', we remove it and drop level
sds <- sds %>%
  filter(gender != "Other")
sds$gender <- droplevels(sds$gender)
# Check table(sds%gender)
table(sds$gender)

## bmi has missing values, use the summary function to confirm. Then fill
# with median as it keeps information and is more robust to outliers compared 
#to mean
summary(sds$bmi)
# median fill the empty entries to preserve information
median_bmi <- median(sds$bmi, na.rm = TRUE)
sds$bmi[is.na(sds$bmi)] <- median_bmi
sum(is.na(sds$bmi))
# confirm summary after median fill
summary(sds$bmi)

# Also remove the id column. It is only a identifier, not a predictor. However,
# it can mislead the models when fitting.
sds <- sds %>% select(-id)
# confirm
names(sds)

# EXPLORATORY DATA ANALYSIS (EDA)
# EDA of the response variable 

sds %>%
  count(stroke)
sds%>%
  tabyl(stroke)

sds %>%
  count(stroke) %>%                                                                  
  ggplot(aes(x = stroke, y = n, fill = stroke)) +                                     
  geom_col(colour = "black") +                                                      
  labs(title = "Distribution of Stroke Outcome",                                     
       x = "Stroke (No/Yes)",                                               
       y = "Count") +                                                               
  scale_fill_manual(values = c("lightblue", "lightpink")) +                         
  theme_light()

# EDA of Categorical Predictors
# contingency tables to examine relationship between response variable and
# categorical predictors

# gender by Stroke
sds%>%
  tabyl(gender, stroke)

sds %>%
  tabyl(gender, stroke) %>%
  adorn_percentages("row") %>%
  adorn_pct_formatting(digits = 2)

# hypertension by Stroke
sds%>%
  tabyl(hypertension, stroke)

sds %>%
  tabyl(hypertension, stroke) %>%
  adorn_percentages("row") %>%
  adorn_pct_formatting(digits = 2)

# heart_disease by Stroke
sds%>%
  tabyl(heart_disease, stroke)

sds %>%
  tabyl(heart_disease, stroke) %>%
  adorn_percentages("row") %>%
  adorn_pct_formatting(digits = 2)

# ever_married by Stroke
sds%>%
  tabyl(ever_married, stroke)

sds %>%
  tabyl(ever_married, stroke) %>%
  adorn_percentages("row") %>%
  adorn_pct_formatting(digits = 2)

# work_type by Stroke
sds%>%
  tabyl(work_type, stroke)

sds %>%
  tabyl(work_type, stroke) %>%
  adorn_percentages("row") %>%
  adorn_pct_formatting(digits = 2)

# Residence_type by Stroke
sds%>%
  tabyl(Residence_type, stroke)

sds %>%
  tabyl(Residence_type, stroke) %>%
  adorn_percentages("row") %>%
  adorn_pct_formatting(digits = 2)

# smoking_status by Stroke
sds%>%
  tabyl(smoking_status, stroke)

sds %>%
  tabyl(smoking_status, stroke) %>%
  adorn_percentages("row") %>%
  adorn_pct_formatting(digits = 2)

# Use stacked bar charts to visualise patternss 

# # Gender by Stroke bar plot
sds %>%
  ggplot(aes(x = gender, fill = stroke)) +
  geom_bar(position = "fill", colour = "black") +
  labs(
    title = "Stroke Outcome by Gender",
    x = "Gender",
    y = "Proportion"
  ) +
  scale_fill_manual(values = c("No" = "lightblue", "Yes" = "lightpink")) +
  scale_y_continuous(labels = scales::percent) +
  theme_light()

# Hypertension by Stroke bar plot
sds %>%
  ggplot(aes(x = hypertension, fill = stroke)) +
  geom_bar(position = "fill", colour = "black") +
  labs(
    title = "Stroke Outcome by Hypertension",
    x = "Hypertension (0 = No, 1 = Yes)",
    y = "Proportion"
  ) +
  scale_fill_manual(values = c("No" = "lightblue", "Yes" = "lightpink")) +
  scale_y_continuous(labels = scales::percent) +
  theme_light()

# Heart Disease by Stroke bar plot
sds %>%
  ggplot(aes(x = heart_disease, fill = stroke)) +
  geom_bar(position = "fill", colour = "black") +
  labs(
    title = "Stroke Outcome by Heart Disease",
    x = "Heart Disease (0 = No, 1 = Yes)",
    y = "Proportion"
  ) +
  scale_fill_manual(values = c("No" = "lightblue", "Yes" = "lightpink")) +
  scale_y_continuous(labels = scales::percent) +
  theme_light()

# Marital Status by Stroke bar plot
sds %>%
  ggplot(aes(x = ever_married, fill = stroke)) +
  geom_bar(position = "fill", colour = "black") +
  labs(
    title = "Stroke Outcome by Marital Status",
    x = "Ever Married",
    y = "Proportion"
  ) +
  scale_fill_manual(values = c("No" = "lightblue", "Yes" = "lightpink")) +
  scale_y_continuous(labels = scales::percent) +
  theme_light()

# Type of Work by Stroke bar plot
sds %>%
  ggplot(aes(x = work_type, fill = stroke)) +
  geom_bar(position = "fill", colour = "black") +
  labs(
    title = "Stroke Outcome by Work Type",
    x = "Work Type",
    y = "Proportion"
  ) +
  scale_fill_manual(values = c("No" = "lightblue", "Yes" = "lightpink")) +
  scale_y_continuous(labels = scales::percent) +
  theme_light()

# Residence Type by Stroke bar plot
sds %>%
  ggplot(aes(x = Residence_type, fill = stroke)) +
  geom_bar(position = "fill", colour = "black") +
  labs(
    title = "Stroke Outcome by Residence Type",
    x = "Residence Type",
    y = "Proportion"
  ) +
  scale_fill_manual(values = c("No" = "lightblue", "Yes" = "lightpink")) +
  scale_y_continuous(labels = scales::percent) +
  theme_light()

# Numerical predictors response variable relationship
summary(sds$age)
summary(sds$avg_glucose_level)
summary(sds$bmi)


# Box plot for age vs stroke outcome
sds %>%
  ggplot(aes(x = stroke, y = age, fill = stroke)) +                                    
  geom_boxplot() +                                                                   
  labs(title = "Age at Stroke Outcome",                                 
       x = "Stroke (No, Yes)",                                               
       y = "Age") +                                                                 
  scale_fill_manual(values = c("No" = "lightblue", "Yes" = "lightpink")) +                        
  theme_minimal()

# Box plot for average glucose level vs Stroke Outcome
sds %>%
  ggplot(aes(x = stroke, y = avg_glucose_level, fill = stroke)) +                                    
  geom_boxplot() +                                                                   
  labs(title = "Average Glucose Level at Stroke Outcome",                                 
       x = "Stroke (No, Yes)",                                               
       y = "Average Glucose Level") +                                                                 
  scale_fill_manual(values = c("No" = "lightblue", "Yes" = "lightpink")) +                        
  theme_minimal()

# Box Plot for BMI vs Stroke 
sds %>%
  ggplot(aes(x = stroke, y = bmi, fill = stroke)) +                                    
  geom_boxplot() +                                                                   
  labs(title = "BMI at Stroke Outcome",                                 
       x = "Stroke (No, Yes)",                                               
       y = "BMI") +                                                                 
  scale_fill_manual(values = c("No" = "lightblue", "Yes" = "lightpink")) +                        
  theme_minimal()
#
# Train/Test split. Using a stratified split helps preserve the class 
# distribution in the training and test sets 

# load package caret for data splitting
library(caret)

# 80/20 stratified split and a seed is set for reproducibility
set.seed(15)

train_index <- caret::createDataPartition( y = sds$stroke,
                                           p = 0.8,
                                           list = FALSE)
train_data <- sds[train_index, ]                                         
test_data  <- sds[-train_index, ]

#Check the dimensions in each set
dim(train_data)
dim(test_data)

#check the class balance is intact
prop.table(table(sds$stroke))
prop.table(table(train_data$stroke))
prop.table(table(test_data$stroke))



# Model Fitting 

# Logistic Regression (GLM)

# Define a single model formula object which includes all predictors 
model_formula <- stroke ~ gender + age + hypertension + heart_disease + 
  ever_married + work_type + Residence_type + avg_glucose_level +
  bmi + smoking_status

fit_logistic <- glm(model_formula,
                          data = train_data,
                          family = binomial)
#shows the summary statistics of the fitted model
summary(fit_logistic)

# odd ratios for coefficients 
exp(coef(fit_logistic))

# odds ration and 95% confidence intervals
logistic_OR <- exp(coef(fit_logistic))
logistic_CI <- exp(confint.default(fit_logistic))
logistic_OR_table <- cbind( OR = logistic_OR,
  CI_low  = logistic_CI[, 1],
  CI_high = logistic_CI[, 2])

# displays odds ratios and confidence intervals
logistic_OR_table

# predicted probabilities on test data
logistic.probs <- predict(fit_logistic,
                     newdata = test_data,
                     type    = "response")
# Converts predicted probabilities into classes using 0.5 cutoff
logistic.pred <- ifelse(logistic.probs > 0.5, "Yes", "No")
logistic.pred <- factor(logistic.pred, levels = levels(train_data$stroke))

# Creates and prints confusion matrix
logistic.cm <- table(Predicted = logistic.pred, Actual = test_data$stroke)
logistic.cm

# Calculates and prints test set accuracy 
logistic.accuracy <- mean(logistic.pred == test_data$stroke)
logistic.accuracy

# Stroke is imbalanced, ROC/AUC
library(pROC)

# roc curve using test set
roc_logistic <- roc(
  response  = test_data$stroke,
  predictor = logistic.probs, levels = c("No", "Yes"), direction = "<")

# calculates and displays area under the roc curve
auc_logistic <- auc(roc_logistic)
auc_logistic

# displays the roc curve
plot(roc_logistic,
     legacy.axes = TRUE,
     main = "ROC Curve Logistic Regression (GLM)",
     col = "blue")
abline(a = 0, b = 1, lty = 5, col = "grey30")

# LASSO Logistic Regression 

library(glmnet)

# creates matrix for training and test data using model_formula
x_train <- model.matrix(model_formula, train_data)[, -1]
x_test <- model.matrix(model_formula, test_data)[, -1]

# Creates numeric response
y_train <- ifelse(train_data$stroke == "Yes", 1, 0)
y_test <- ifelse(test_data$stroke == "Yes", 1, 0)

# Set seed for Lasso cross validation for reproducibility
set.seed(15)

# Cross validated LASSO, alpha = 1
cv_lasso <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 1)

# Plot cross validation curve
plot(cv_lasso)

# Best lambda chosen 
cv_lasso$lambda.min
cv_lasso$lambda.1se

# Fit Lasso model on training data at lambda.min
lasso_fit <- glmnet(
  x      = x_train,
  y      = y_train,
  family = "binomial",
  alpha  = 1,
  lambda = cv_lasso$lambda.min)

# Predict probabilities of stroke on test data
lasso.probs <- predict(
  lasso_fit,
  newx = x_test,
  type = "response")

# Converts predicted probabilities into classes using 0.5 cutoff
lasso.pred <- ifelse(lasso.probs > 0.5, "Yes", "No")
lasso.pred <- factor(lasso.pred, levels = levels(train_data$stroke))

# Creates and prints confusion matrix
lasso.cm <- table(Predicted = lasso.pred, Actual = test_data$stroke)
lasso.cm

# Calculates and prints test set accuracy 
lasso.accuracy <- mean(lasso.pred == test_data$stroke)
lasso.accuracy

# roc curve using test set
roc_lasso <- roc(
  response  = test_data$stroke, predictor = as.numeric(lasso.probs),
  levels    = c("No", "Yes"), direction = "<")

# calculates and displays area under the roc curve
auc_lasso <- auc(roc_lasso)
auc_lasso

# displays the roc curve
plot( roc_lasso,
  legacy.axes = TRUE,
  main = "ROC Curve LASSO Logistic Regression",
  col  = "red")
abline(a = 0, b = 1, lty = 5, col = "grey30")

# Try a lower cutoff due to strong class imbalance, a cutoff below 0.5 was 
# considered. Renamed for clarity
cutoff_lower <- 0.05

lasso.pred_lower <- ifelse(lasso.probs > cutoff_lower, "Yes", "No")                               
lasso.pred_lower <- factor(lasso.pred_lower, levels = levels(train_data$stroke))              

lasso.cm_lower <- table(Predicted = lasso.pred_lower, Actual = test_data$stroke)
lasso.cm_lower

lasso.accuracy_lower <- mean(lasso.pred_lower == test_data$stroke)
lasso.accuracy_lower


# Random Forest

library(randomForest)

# Set seed for reproducibility
set.seed(15)

# Fit a random forest on the training data
rf_fit <- randomForest(
  stroke ~ .,
  data = train_data,
  ntree = 500, 
  mtry = 3,                                                                      
  importance = TRUE)

# Prints random forest 
rf_fit

# Plot of importance measures 
varImpPlot(rf_fit, main = "Random Forest Variable Importance")

# Predict probabilities of stroke of test set
rf_probs <- predict(rf_fit, newdata = test_data, type = "prob") [, "Yes"]

# Creates and prints confusion matrix
rf_pred <- ifelse(rf_probs > 0.5, "Yes", "No")
rf_pred <- factor(rf_pred, levels = levels(train_data$stroke))

rf_cm <- table(Predicted = rf_pred, Actual = test_data$stroke)
rf_cm

# Test accuracy for random forest
rf_accuracy <- mean(rf_pred == test_data$stroke)
rf_accuracy

# roc curve using test set
roc_rf <- roc(
  response = test_data$stroke, predictor = rf_probs,
  levels = c("No", "Yes"), direction = "<")

# calculates and displays area under the roc curve
auc_rf <- auc(roc_rf)
auc_rf

# displays the roc curve
plot(
  roc_rf,
  legacy.axes = TRUE,
  main = "ROC Curve Random Forest",
  col  = "darkgreen")
abline(a = 0, b = 1, lty = 5, col = "grey30")

# Try a lower cutoff due to strong class imbalance, a cutoff below 0.5 was 
# considered. Renamed for clarity

rf_pred_lower <- ifelse(rf_probs > cutoff_lower, "Yes", "No")
rf_pred_lower <- factor(rf_pred_lower, levels = levels(train_data$stroke))                     

rf_cm_lower <- table(Predicted = rf_pred_lower, Actual = test_data$stroke)
rf_cm_lower

rf_accuracy_lower <- mean(rf_pred_lower == test_data$stroke)
rf_accuracy_lower

# Feed-Forward Neural Network
library(h2o)

# Connect to h2o server
h2o.init()

# Convert training and test data to h2o data
train_hex <- as.h2o(train_data)
test_hex <- as.h2o(test_data)

# Designate response variable and remaining predictor variables
y <- "stroke"
x <- setdiff(names(train_hex), y)

# Converts response variable to a factor inside h2o
train_hex[, y] <- as.factor(train_hex[, y])
test_hex [, y] <- as.factor(test_hex[, y])


# Fit the deep learning Feed-Forward model 
dlff_model <- h2o.deeplearning(
  x = x,
  y = y,
  training_frame = train_hex,
  activation = "Rectifier",
  hidden = c(16, 8),
  epochs = 50,
  seed = 15)

# Model summary
dlff_model

# Predicts probabilities on test set
dlff_pred <- h2o.predict(dlff_model, test_hex)

# Probabilities of "yes"
dlff_probs <- as.vector(dlff_pred[,"Yes"])
colnames(dlff_pred)
summary(dlff_probs)

# Creates and prints confusion matrix
dlff_pred_05 <- ifelse(dlff_probs > 0.5, "Yes", "No")
dlff_pred_05 <- factor(dlff_pred_05, levels = levels(train_data$stroke))
table(Predicted = dlff_pred_05, Actual = test_data$stroke)

# Test accuracy for Feed-Forward Neural Network
dlff_accuracy <- mean(dlff_pred_05== test_data$stroke)
dlff_accuracy

# roc curve using test set
roc_dlff <- roc(
  response  = test_data$stroke,
  predictor = dlff_probs,
  levels    = c("No", "Yes"),
  direction = "<")

# calculates and displays area under the roc curve
auc_dlff <- auc(roc_dlff)
auc_dlff

# displays the roc curve
plot(roc_dlff,
     legacy.axes = TRUE,
     main = "ROC Curve Feed-forward Neural Network (h2o)",
     col = "purple")
abline(a = 0, b = 1, lty = 5, col = "grey30")

# Try a lower cutoff due to strong class imbalance, a cutoff below 0.5 was 
# considered. Renamed for clarity
cutoff_lower <- 0.05
dlff_pred_lower <- ifelse(dlff_probs > cutoff_lower, "Yes", "No")
dlff_pred_lower <- factor(dlff_pred_lower, levels = levels(train_data$stroke))

table(Predicted = dlff_pred_lower, Actual = test_data$stroke)

dlff_accuracy_lower <- mean(dlff_pred_lower == test_data$stroke)
dlff_accuracy_lower

# Model Comparison. (GLM vs LASSO vs Random Forest vs Feed-Forward Neural 
# Network)

# Converts the test outcome into a factor for comparison tables
true <- factor(test_data$stroke, levels = c("No", "Yes"))

# Ensures probabilities are numeric 
logistic.probs <- as.numeric(logistic.probs)
lasso.probs <- as.numeric(lasso.probs)
rf_probs <- as.numeric(rf_probs)
dlff_probs <- as.numeric(dlff_probs)

# Cutoff for confusion matrix comparison
cutoff <-0.5

# Creates predicted classes and confusion matrix for Logistic Regression model 
logistic.pred <- ifelse(logistic.probs > cutoff, "Yes", "No")
logistic.pred <- factor(logistic.pred, levels = levels(true))
logistic.cm   <- table(Predicted = logistic.pred, Actual = true)

# Creates predicted classes and confusion matrix for LASSO model
lasso.pred <- ifelse(lasso.probs > cutoff, "Yes", "No") 
lasso.pred <- factor(lasso.pred, levels = levels(true))
lasso.cm   <- table(Predicted = lasso.pred, Actual = true)

# Creates predicted classes and confusion matrix for Random Forest model
rf_pred <- ifelse(rf_probs > cutoff, "Yes", "No")
rf_pred <- factor(rf_pred, levels = levels(true))                                  
rf_cm   <- table(Predicted = rf_pred, Actual = true)

# Creates predicted classes and confusion matrix for Feed-forward model
dlff_pred <- ifelse(dlff_probs > cutoff, "Yes", "No")                               
dlff_pred <- factor(dlff_pred, levels = levels(true))                              
dlff_cm   <- table(Predicted = dlff_pred, Actual = true)

# Prints confusion matrices
logistic.cm                                                                             
lasso.cm                                                                           
rf_cm                                                                              
dlff_cm

# Defines function to compute metrics from confusion matrix
calc_metrics <- function(cm) {
  all_levels <- c("No", "Yes")                                                      
  cm_full <- matrix(0,
                    nrow = 2, ncol = 2,                                             
                    dimnames = list(Predicted = all_levels,
                                    Actual    = all_levels))
  cm_full[rownames(cm), colnames(cm)] <- cm                                         
  cm <- cm_full                                                                     
  
  TN <- cm["No",  "No"]                                                             
  FP <- cm["Yes", "No"]                                                             
  FN <- cm["No",  "Yes"]                                                            
  TP <- cm["Yes", "Yes"]                                                            
  
  accuracy    <- (TP + TN) / sum(cm)                                                
  sensitivity <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)                          
  specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)                          
  precision   <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)                          
  f1   <- ifelse(!is.na(precision) & !is.na(sensitivity) & 
                   (precision + sensitivity) > 0, 
                 2 * (precision * sensitivity) / (precision + sensitivity), NA)
  
list(
    accuracy    = as.numeric(accuracy),
    sensitivity = as.numeric(sensitivity),
    specificity = as.numeric(specificity),
    precision   = as.numeric(precision),
    f1          = as.numeric(f1))}                                                                                   

# Computes performance metrics for each model's confusion matrix
logistic.metrics  <- calc_metrics(logistic.cm)                                                
lasso.metrics     <- calc_metrics(lasso.cm)                                             
rf_metrics        <- calc_metrics(rf_cm)                                                 
dlff_metrics      <- calc_metrics(dlff_cm)

# Computes Roc/Auc for each model

roc_logistic   <- roc(response = true, predictor = logistic.probs, 
                      levels = c("No","Yes"), direction = "<")
roc_lasso      <- roc(response = true, predictor = lasso.probs,
                      levels = c("No","Yes"), direction = "<")
roc_rf         <- roc(response = true, predictor = rf_probs,
                      levels = c("No","Yes"), direction = "<")
roc_dlff       <- roc(response = true, predictor = dlff_probs,
                      levels = c("No","Yes"), direction = "<")

# Creates tibble summerising metrics for each model
comparison <- tibble(                                                               
  Model = c("Logistic Regression (GLM)",
            "LASSO Logistic Regression",
            "Random Forest",
            "Feed-forward Neural Network (h2o)"),
  
Cutoff = cutoff,
  
AUC =         c(as.numeric(pROC::auc(roc_logistic)),
                as.numeric(pROC::auc(roc_lasso)),
                as.numeric(pROC::auc(roc_rf)),
                as.numeric(pROC::auc(roc_dlff))),
                  
Accuracy =    c(logistic.metrics$accuracy,
                lasso.metrics$accuracy,
                rf_metrics$accuracy,
                dlff_metrics$accuracy),

Sensitivity = c(logistic.metrics$sensitivity,
                lasso.metrics$sensitivity,
                rf_metrics$sensitivity,
                dlff_metrics$sensitivity),

Specificity = c(logistic.metrics$specificity,
                lasso.metrics$specificity,
                rf_metrics$specificity,
                dlff_metrics$specificity),

Precision =   c(logistic.metrics$precision,
                lasso.metrics$precision,
                rf_metrics$precision,
                dlff_metrics$precision),

F1 =           c(logistic.metrics$f1,
                 lasso.metrics$f1,
                 rf_metrics$f1,
                 dlff_metrics$f1)
)%>%                                                                               
mutate(across(where(is.numeric), ~ round(., 4)))                                  

# Prints model comparison
comparison                                                                          


# Selects "best" model by AUC 

comparison %>%
  arrange(desc(AUC)) %>%                                                            
  slice(1)                                                                          




