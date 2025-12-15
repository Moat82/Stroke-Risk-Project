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

