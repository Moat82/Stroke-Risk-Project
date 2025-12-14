library(dplyr)
library(janitor)
library(ggplot2)
library(tibble)
# read healthcare_dataset_stroke_csv and assign it to the object 'sds' for ease 
# of use. Converts N/A to missing entries 
sds <- read.csv("healthcare_dataset_stroke_data.csv", header = TRUE,
                na.strings = c("N/A", ""))
