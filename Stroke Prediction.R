###############################################################################
## Stroke Risk Prediction Using Statistical and Machine Learning Models       #
###############################################################################

## PACKAGE AND DATA SET INPUT #################################################
## Required packages for data manipulation, modelling, and evaluation         #
library(dplyr) 
library(tibble)
library(ggplot2) 
library(janitor)                                                                    
library(caret)                                                                 
library(glmnet)                                                                 
library(randomForest)  
library(pROC)                                                                       
library(h2o)                                                                        

## read healthcare_data set_stroke_csv. Assign it to the object 'sds'(stroke  #
## data set), and converts N/A to missing entries                             #
sds <- read.csv("healthcare_dataset_stroke_data.csv", header = TRUE,
                na.strings = c("N/A", ""))

## DATA PREPERATION AND DATA QUALITY CHECK ####################################
## Converting the response variable and the categorical predictors to factors #
## ensures R implements them correctly during model fitting                   #
sds <- sds%>%
  mutate(
    stroke = factor(stroke, levels = c(0, 1), labels = c("No", "Yes")),            
    gender = factor(gender),
    hypertension = factor(hypertension),
    heart_disease = factor(heart_disease),
    ever_married = factor(ever_married),
    work_type = factor(work_type),
    Residence_type = factor(Residence_type),
    smoking_status = factor(smoking_status))

## Data quality check to asses the structure of data by inspecting variable   #
## names, dimensions, and any missing or invalid entries                      #
names(sds)
dim(sds)
str(sds)
sum(is.na(sds))
colSums(is.na(sds))

## table() counts the frequency of each category in a variable                #
table(sds$gender)
table(sds$ever_married)
table(sds$work_type)
table(sds$Residence_type)
table(sds$smoking_status)

## The following code allows us to remove the "other" from the predictor      #
## gender and drops the unused factor level                                   #
sds <- sds %>%
  filter(gender != "Other")
sds$gender <- droplevels(sds$gender)

## Check table(sds%gender) to confirm the level was removed
table(sds$gender)

## bmi has missing values, use the summary() to confirm, then impute with the #
## median since it is more robust to outliers than the mean                   #
summary(sds$bmi)

## Impute missing bmi values with the median                                  #
median_bmi <- median(sds$bmi, na.rm = TRUE)
sds$bmi[is.na(sds$bmi)] <- median_bmi

## Confirm no missing entries remain                                          #
summary(sds$bmi)

## Remove the id column since it is an identifier, not a predictor, and may   #
## mislead the model fitting                                                  #
sds <- sds %>% select(-id)

## Confirm the column removal                                                 #
names(sds)

## EXPLORATORY DATA ANALYSIS (EDA) ############################################
## EDA is used to detect possible errors, understand the structure of the     #
## data set, examine the patterns, distributions, and relationships between   #
## the response variable and the predictor variables                          #
###############################################################################

## Response variable EDA: counts and distribution bar plot                    #                    
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

## EDA of categorical predictors: contingency tables and bar plots show the   #
## relationship between response variable and categorical predictors          #

# gender by stroke contingency tables                                         #
sds%>%
  tabyl(gender, stroke)

sds %>%
  tabyl(gender, stroke) %>%
  adorn_percentages("row") %>%
  adorn_pct_formatting(digits = 2)

# gender by stroke bar plot                                                   #
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

# hypertension by stroke contingency tables                                   #
sds%>%
  tabyl(hypertension, stroke)

sds %>%
  tabyl(hypertension, stroke) %>%
  adorn_percentages("row") %>%
  adorn_pct_formatting(digits = 2)

# hypertension by stroke bar plot                                             #
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

# heart_disease by stroke contingency tables                                  #
sds%>%
  tabyl(heart_disease, stroke)

sds %>%
  tabyl(heart_disease, stroke) %>%
  adorn_percentages("row") %>%
  adorn_pct_formatting(digits = 2)

# Heart Disease by stroke bar plot                                            #
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

# ever_married by stroke contingency tables                                   #
sds%>%
  tabyl(ever_married, stroke)

sds %>%
  tabyl(ever_married, stroke) %>%
  adorn_percentages("row") %>%
  adorn_pct_formatting(digits = 2)

# ever_married by Stroke bar plot                                             #
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

# work_type by stroke contingency tables                                      #
sds%>%
  tabyl(work_type, stroke)

sds %>%
  tabyl(work_type, stroke) %>%
  adorn_percentages("row") %>%
  adorn_pct_formatting(digits = 2)

# work_type by stroke bar plot                                                #
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

# Residence_type by stroke contingency tables                                 #
sds%>%
  tabyl(Residence_type, stroke)

sds %>%
  tabyl(Residence_type, stroke) %>%
  adorn_percentages("row") %>%
  adorn_pct_formatting(digits = 2)

# Residence_type by stroke bar plot                                           #
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

# smoking_status by stroke contingency tables                                 #
sds%>%
  tabyl(smoking_status, stroke)

sds %>%
  tabyl(smoking_status, stroke) %>%
  adorn_percentages("row") %>%
  adorn_pct_formatting(digits = 2)

# smoking_status by stroke bar plot                                           #   
sds %>%
  ggplot(aes(x = smoking_status, fill = stroke)) +
  geom_bar(position = "fill", colour = "black") +
  labs(
    title = "Stroke Outcome by Smoking Status",
    x = "Smoking Status",
    y = "Proportion"
  ) +
  scale_fill_manual(values = c("No" = "lightblue", "Yes" = "lightpink")) +
  scale_y_continuous(labels = scales::percent) +
  theme_light()

## EDA of numerical predictors: summary statistics and box plots show the     #
## relationship between response variable and the numerical predictors        #

# summary statistics for age by stroke                                        # 
summary(sds$age)

# box plot for age vs stroke                                                  #
sds %>%
  ggplot(aes(x = stroke, y = age, fill = stroke)) +                                    
  geom_boxplot() +                                                                   
  labs(title = "Age at Stroke Outcome",                                 
       x = "Stroke (No, Yes)",                                               
       y = "Age") +                                                                 
  scale_fill_manual(values = c("No" = "lightblue", "Yes" = "lightpink")) +                        
  theme_minimal()

# summary statistics fro average glucose levels by stroke                     #
summary(sds$avg_glucose_level)

# box plot for average glucose level by stroke                                #
sds %>%
  ggplot(aes(x = stroke, y = avg_glucose_level, fill = stroke)) +                                    
  geom_boxplot() +                                                                   
  labs(title = "Average Glucose Level at Stroke Outcome",                                 
       x = "Stroke (No, Yes)",                                               
       y = "Average Glucose Level") +                                                                 
  scale_fill_manual(values = c("No" = "lightblue", "Yes" = "lightpink")) +                        
  theme_minimal()

# summary statistics for bmi by stroke                                        #
summary(sds$bmi)

# box plot for bmi by stroke                                                  #
sds %>%
  ggplot(aes(x = stroke, y = bmi, fill = stroke)) +                                    
  geom_boxplot() +                                                                   
  labs(title = "BMI at Stroke Outcome",                                 
       x = "Stroke (No, Yes)",                                               
       y = "BMI") +                                                                 
  scale_fill_manual(values = c("No" = "lightblue", "Yes" = "lightpink")) +                        
  theme_minimal()

## Stratified train/test split ################################################
## Using a stratified split of 80/20 helps preserve the class balance and a   #
## fixed seed ensures reproducibilty                                          #


# 80/20 stratified split and a seed is set for reproducibility                #
set.seed(15)

train_index <- caret::createDataPartition( 
  y = sds$stroke,
  p = 0.8,
  list = FALSE)

train_data <- sds[train_index, ]                                         
test_data  <- sds[-train_index, ]

# verify training and test set dimension                                      #
dim(train_data)
dim(test_data)

# confirm class balance is intact after stratified split                      #
prop.table(table(sds$stroke))
prop.table(table(train_data$stroke))
prop.table(table(test_data$stroke))



## Model Fitting ##############################################################

## define a single model formula object which includes all predictors         #
model_formula <- stroke ~ gender + age + hypertension + heart_disease + 
  ever_married + work_type + Residence_type + avg_glucose_level +
  bmi + smoking_status


## Logistic regression (GLM) baseline model ###################################
glm_model <- glm(model_formula,
                          data = train_data,
                          family = binomial)

# model summary and coefficient significance                                  #
summary(glm_model)

# odd ratios for coefficients 
exp(coef(glm_model))

# odds ratios and 95% confidence intervals                                    #
glm_OR <- exp(coef(glm_model))
glm_CI <- exp(confint.default(glm_model))
glm_OR_table <- cbind( OR = glm_OR,
  CI_low  = glm_CI[, 1],
  CI_high = glm_CI[, 2])

glm_OR_table

# predicted probabilities on test set                                         #
glm_probs <- predict(glm_model,
                     newdata = test_data,
                     type    = "response")

# converts predicted probabilities into classes using 0.5 cutoff              #
glm_pred <- ifelse(glm_probs > 0.5, "Yes", "No")
glm_pred <- factor(glm_pred, levels = levels(train_data$stroke))

# test set confusion matrix and accuracy                                      #
glm_cm <- table(Predicted = glm_pred, Actual = test_data$stroke)
glm_cm

glm_accuracy <- mean(glm_pred == test_data$stroke)
glm_accuracy

# ROC curve and ACU on the test set                                           # 
roc_glm <- roc(
  response  = test_data$stroke,
  predictor = glm_probs, 
  levels = c("No", "Yes"), 
  direction = "<")

# area under the ROC curve                                                    #
auc_glm<- auc(roc_glm)
auc_glm

# plot ROC curve
plot(roc_glm,
     legacy.axes = TRUE,
     main = "Logistic Regression (GLM) ROC Curve",
     col = "blue")
abline(a = 0, b = 1, lty = 5, col = "grey30")

## LASSO logistic regression (L1 penalty) #####################################                                  

# matrix for LASSO regression                                                 #
x_train <- model.matrix(model_formula, train_data)[, -1]
x_test <- model.matrix(model_formula, test_data)[, -1]

# creates numeric response                                                    #
y_train <- ifelse(train_data$stroke == "Yes", 1, 0)
y_test <- ifelse(test_data$stroke == "Yes", 1, 0)

# Set seed for Lasso cross validation for reproducibility                     #
set.seed(15)

# cross validated LASSO, alpha = 1                                            #
cv_lasso <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 1)

# plot cross validation curve                                                 #
plot(cv_lasso)

# optimal lambda chosen                                                       #
cv_lasso$lambda.min
cv_lasso$lambda.1se

# fit Lasso model at lambda.min                                               #
lasso_fit <- glmnet(
  x      = x_train,
  y      = y_train,
  family = "binomial",
  alpha  = 1,
  lambda = cv_lasso$lambda.min)

# predicted probabilities of stroke on test data                              #
lasso_probs <- predict(
  lasso_fit,
  newx = x_test,
  type = "response")

# converts predicted probabilities into classes using 0.5 cutoff              #
lasso_pred <- ifelse(lasso_probs > 0.5, "Yes", "No")
lasso_pred <- factor(lasso_pred, levels = levels(train_data$stroke))

# creates confusion matrix and accuracy                                       #
lasso_cm <- table(Predicted = lasso_pred, Actual = test_data$stroke)
lasso_cm

lasso_accuracy <- mean(lasso_pred == test_data$stroke)
lasso_accuracy

# ROC curve on test set                                                       #
roc_lasso <- roc(
  response  = test_data$stroke, predictor = as.numeric(lasso_probs),
  levels    = c("No", "Yes"), direction = "<")

# AUC of test set                                                             #
auc_lasso <- auc(roc_lasso)
auc_lasso

# plots ROC curve                                                             #
plot( roc_lasso,
  legacy.axes = TRUE,
  main = "LASSO Logistic Regression ROC Curve",
  col  = "red")
abline(a = 0, b = 1, lty = 5, col = "grey30")

## Random forest ##############################################################

set.seed(15)

# fit a random forest on the training set                                     #
rf_fit <- randomForest(
  stroke ~ .,
  data = train_data,
  ntree = 500, 
  mtry = 3,                                                                      
  importance = TRUE)

rf_fit

# plot of importance measures 
varImpPlot(rf_fit, main = "Random Forest Variable Importance")

# predicted probabilities of stroke of test set                               #
rf_probs <- predict(rf_fit, newdata = test_data, type = "prob") [, "Yes"]

# converts predicted probabilities into classes using 0.5 cutoff              # 
rf_pred <- ifelse(rf_probs > 0.5, "Yes", "No")
rf_pred <- factor(rf_pred, levels = levels(train_data$stroke))

# confusion matrix and accuracy                                               #
rf_cm <- table(Predicted = rf_pred, Actual = test_data$stroke)
rf_cm

rf_accuracy <- mean(rf_pred == test_data$stroke)
rf_accuracy

# ROC curve and AUC on test set                                               #
roc_rf <- roc(
  response = test_data$stroke, predictor = rf_probs,
  levels = c("No", "Yes"), direction = "<")

auc_rf <- auc(roc_rf)
auc_rf

# plots ROC curve                                                             #              
plot(
  roc_rf,
  legacy.axes = TRUE,
  main = "ROC Curve Random Forest",
  col  = "darkgreen")
abline(a = 0, b = 1, lty = 5, col = "grey30")

## Feed-forward neural network (h2o) ##########################################

# connect to h2o server                                                       #
h2o.init()

# convert data to h2o frames                                                  #       
train_hex <- as.h2o(train_data)
test_hex <- as.h2o(test_data)

# designate response variable and remaining predictor variables               #
y <- "stroke"
x <- setdiff(names(train_hex), y)

# converts response variable to a factor inside h2o                           #
train_hex[, y] <- as.factor(train_hex[, y])
test_hex [, y] <- as.factor(test_hex[, y])


# fit feed-forward model                                                      #
dlff_model <- h2o.deeplearning(
  x = x,
  y = y,
  training_frame = train_hex,
  activation = "Rectifier",
  hidden = c(16, 8),
  epochs = 50,
  seed = 15)

dlff_model

# predicted probabilities on test set                                         #
dlff_pred <- h2o.predict(dlff_model, test_hex)

# probabilities for the positive class                                        #
dlff_probs <- as.vector(dlff_pred[,"Yes"])
colnames(dlff_pred)
summary(dlff_probs)

# converts probabilities to classes using 0.5 cutoff                          #
dlff_pred <- ifelse(dlff_probs > 0.5, "Yes", "No")
dlff_pred <- factor(dlff_pred, levels = levels(train_data$stroke))

# confusion matrix and accuracy                                               #
dlff_cm <- table(Predicted = dlff_pred, Actual = test_data$stroke)
dlff_cm

dlff_accuracy <- mean(dlff_pred == test_data$stroke)
dlff_accuracy

# ROC and AUC on test set.                                                    #
roc_dlff <- roc(
  response  = test_data$stroke,
  predictor = dlff_probs,
  levels    = c("No", "Yes"),
  direction = "<")

auc_dlff <- auc(roc_dlff)
auc_dlff

# plot ROC curve
plot(roc_dlff,
     legacy.axes = TRUE,
     main = "Feed-forward Neural Network ROC Curve (h2o)",
     col = "purple")
abline(a = 0, b = 1, lty = 5, col = "grey30")


## MODEL COMPARISION USING ROC/AUC AND CLASSIFICATION METRICS #################

# converts the test outcome into a factor for comparison tables               #
true <- factor(test_data$stroke, levels = c("No", "Yes"))

# ensures probabilities are numeric 
glm_probs <- as.numeric(glm_probs)
lasso_probs <- as.numeric(lasso_probs)
rf_probs <- as.numeric(rf_probs)
dlff_probs <- as.numeric(dlff_probs)

# cutoff for confusion matrix comparison                                      #
cutoff <-0.5

# predicted classes and confusion matrix for logistic regression model        # 
glm_pred <- ifelse(glm_probs > cutoff, "Yes", "No")
glm_pred <- factor(glm_pred, levels = levels(true))
glm_cm   <- table(Predicted = glm_pred, Actual = true)

# predicted classes and confusion matrix for LASSO model                      #
lasso_pred <- ifelse(lasso_probs > cutoff, "Yes", "No") 
lasso_pred <- factor(lasso_pred, levels = levels(true))
lasso_cm   <- table(Predicted = lasso_pred, Actual = true)

# predicted classes and confusion matrix for random forest model              #
rf_pred <- ifelse(rf_probs > cutoff, "Yes", "No")
rf_pred <- factor(rf_pred, levels = levels(true))                                  
rf_cm   <- table(Predicted = rf_pred, Actual = true)

# predicted classes and confusion matrix for feed-forward model               #
dlff_pred <- ifelse(dlff_probs > cutoff, "Yes", "No")                               
dlff_pred <- factor(dlff_pred, levels = levels(true))                              
dlff_cm   <- table(Predicted = dlff_pred, Actual = true)

# prints confusion matrices at cutoff 0.5                                     #
glm_cm                                                                             
lasso_cm                                                                           
rf_cm                                                                              
dlff_cm

# defines function to compute metrics from confusion matrix                   #
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

# computes performance metrics for each model's confusion matrix              #
glm_metrics  <- calc_metrics(glm_cm)                                                
lasso_metrics     <- calc_metrics(lasso_cm)                                             
rf_metrics        <- calc_metrics(rf_cm)                                                 
dlff_metrics      <- calc_metrics(dlff_cm)

# ROC/AUC comparison for all models                                           #

roc_glm        <- roc(response = true, predictor = glm_probs, 
                      levels = c("No","Yes"), direction = "<")
roc_lasso      <- roc(response = true, predictor = lasso_probs,
                      levels = c("No","Yes"), direction = "<")
roc_rf         <- roc(response = true, predictor = rf_probs,
                      levels = c("No","Yes"), direction = "<")
roc_dlff       <- roc(response = true, predictor = dlff_probs,
                      levels = c("No","Yes"), direction = "<")

# tibble summerising metrics for each model                                   #
comparison <- tibble(                                                               
  Model = c("Logistic Regression (GLM)",
            "LASSO Logistic Regression",
            "Random Forest",
            "Feed-forward Neural Network (h2o)"),
  
Cutoff = cutoff,
  
AUC = c(
  as.numeric(pROC::auc(roc_glm)),
  as.numeric(pROC::auc(roc_lasso)),
  as.numeric(pROC::auc(roc_rf)),
  as.numeric(pROC::auc(roc_dlff))),

Accuracy = c(
  glm_metrics$accuracy,
  lasso_metrics$accuracy,
  rf_metrics$accuracy,
  dlff_metrics$accuracy),

Sensitivity = c(
  glm_metrics$sensitivity,
  lasso_metrics$sensitivity,
  rf_metrics$sensitivity,
  dlff_metrics$sensitivity),

Specificity = c(
  glm_metrics$specificity,
  lasso_metrics$specificity,
  rf_metrics$specificity,
  dlff_metrics$specificity),

Precision = c(
  glm_metrics$precision,
  lasso_metrics$precision,
  rf_metrics$precision,
  dlff_metrics$precision),

F1 = c(
  glm_metrics$f1,
  lasso_metrics$f1,
  rf_metrics$f1,
  dlff_metrics$f1)
)%>%
  mutate(across(where(is.numeric), ~ round(., 4)))                                  

# prints model comparison                                                     #
comparison                                                                          


# best model by AUC (threshold independent)                                   #
comparison %>%
  arrange(desc(AUC)) %>%                                                            
  slice(1)  

## Threshold comparison #######################################################
## Assesses the impact of different cutoffs on model performance              #

# displays class proportions in training set                                  #
prop.table(table(train_data$stroke))

# computes prevalence of positive class in training set                       #
positive_level <- "Yes"
cutoff_lower <- mean(train_data$stroke == positive_level)

cutoff_lower

# predicted classes and matrix for logistic regression at cutoff_lower        #
glm_pred_lower <- ifelse(glm_probs > cutoff_lower, "Yes", "No")
glm_pred_lower <- factor(glm_pred_lower, levels = levels(true))
glm_cm_lower   <- table(Predicted = glm_pred_lower, Actual = true)

# predicted classes and matrix for LASSO at cutoff_lower                      #
lasso_pred_lower <- ifelse(lasso_probs > cutoff_lower, "Yes", "No")
lasso_pred_lower <- factor(lasso_pred_lower, levels = levels(true))
lasso_cm_lower   <- table(Predicted = lasso_pred_lower, Actual = true)

# predicted classes and matrix for random forest at cutoff_lower              # 
rf_pred_lower <- ifelse(rf_probs > cutoff_lower, "Yes", "No")
rf_pred_lower <- factor(rf_pred_lower, levels = levels(true))
rf_cm_lower <- table(Predicted = rf_pred_lower, Actual = true)

# predicted classes and matrix for feed-forward at cutoff_lower               #  

dlff_pred_lower <- ifelse(dlff_probs > cutoff_lower, "Yes", "No")
dlff_pred_lower <- factor(dlff_pred_lower, levels = levels(true))
dlff_cm_lower   <- table(Predicted = dlff_pred_lower, Actual = true)


# compute metrics from a 2x2 confusion matrix                                 #
calc_metrics_lower <- function(cm) {
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
  
# Computes performance metrics
  accuracy    <- (TP + TN) / sum(cm)
  sensitivity <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)      
  specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)      
  precision   <- ifelse((TP + FP) > 0, TP / (TP + FP), NA)      
  f1          <- ifelse(is.na(precision) | is.na(sensitivity) | 
                          (precision + sensitivity) == 0, NA,
                      2 * precision * sensitivity / (precision + sensitivity))
  
# returns all metrics as a named list                                         #
list(accuracy = accuracy,
     sensitivity = sensitivity,
     specificity = specificity,
     precision = precision,
     f1 = f1)}


# compute metrics at cutoff_lower                                             #
glm_metrics_lower     <- calc_metrics_lower(glm_cm_lower)
lasso_metrics_lower   <- calc_metrics_lower(lasso_cm_lower)
rf_metrics_lower      <- calc_metrics_lower(rf_cm_lower)
dlff_metrics_lower    <- calc_metrics_lower(dlff_cm_lower)


# comparison table for cutoff_lower                                           #
comparison_lower <- tibble(
  Model = c(
    "Logistic Regression (GLM)",
    "LASSO Logistic Regression",
    "Random Forest",
    "Feed-forward Neural Network (h2o)"),
  
Cutoff = cutoff_lower,
  
AUC = c(
    as.numeric(pROC::auc(roc_glm)),
    as.numeric(pROC::auc(roc_lasso)),
    as.numeric(pROC::auc(roc_rf)),
    as.numeric(pROC::auc(roc_dlff))),
  
Accuracy = c(
    glm_metrics_lower$accuracy,
    lasso_metrics_lower$accuracy,
    rf_metrics_lower$accuracy,
    dlff_metrics_lower$accuracy),

Sensitivity = c(
    glm_metrics_lower$sensitivity,
    lasso_metrics_lower$sensitivity,
    rf_metrics_lower$sensitivity,
    dlff_metrics_lower$sensitivity),

Specificity = c(
    glm_metrics_lower$specificity,
    lasso_metrics_lower$specificity,
    rf_metrics_lower$specificity,
    dlff_metrics_lower$specificity),

Precision = c(
    glm_metrics_lower$precision,
    lasso_metrics_lower$precision,
    rf_metrics_lower$precision,
    dlff_metrics_lower$precision),

F1 = c(
    glm_metrics_lower$f1,
    lasso_metrics_lower$f1,
    rf_metrics_lower$f1,
    dlff_metrics_lower$f1)
)%>%
  mutate(across(where(is.numeric), ~ round(., 4)))

# print cutoff_lower comparison table                                         #
comparison_lower

# rank by sensitivity                                                         #
comparison_lower %>%
  arrange(desc(Sensitivity))

# compare default cutoff vs cutoff_lower                                      #
comparison <- comparison %>%
  mutate(Threshold = cutoff)

comparison_lower <- comparison_lower %>%
  mutate(Threshold = cutoff_lower)
comparison_both <- bind_rows(comparison, comparison_lower)
comparison_both

