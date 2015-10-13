library("googlesheets")
library("data.table")
suppressPackageStartupMessages(library("dplyr"))
library("dplyr")
library("reshape2")
library("ggplot2")
install.packages("data.table")

# Fetch data from Google Docs spreadsheet
survey_sheet <- gs_key("1KPmQid_luBPci7puqz8i8IN1nCMYNwYEpe1ke_az0j4")
sentences <- survey_sheet %>% gs_read(ws = "Sentences")

survey_results <- survey_sheet %>% gs_read(ws = "Form Responses 1")
melted_results <- melt(survey_results)
melted_results$variable <- gsub("\\.", " ", melted_results$variable)
colnames(melted_results)[3] <- "Sentence"

# combine the sentences and survey results into one data frame
results <- inner_join(melted_results, sentences)
#results$value = factor(results$value)
results %>% group_by(Model, SmoothingMethod)


# Histogram results per model and smooothing method
ggplot(results, aes(x=value)) +
  facet_wrap(Model~SmoothingMethod) +
  scale_x_discrete(limits=c(1,2,3,4,5)) +
  geom_histogram(binwidth = 1, color="black", fill="white") +
  xlab("Score") +
  ggtitle("Results by model and smoothing method") +
  theme_bw()

# Histogram results per model
ggplot(results, aes(x=value)) +
  facet_wrap(~Model) +
  geom_histogram(alpha=I(0.8), color="black", fill="white") +
  scale_x_discrete(limits=c(1,2,3,4,5)) +
  xlab("Score") +
  ggtitle("Results by model")

# Boxplot results per model
ggplot(results, aes(x=Model, y=value)) +
  #facet_wrap(~Model) +
  geom_boxplot() +
  xlab("Model") + 
  ylab("Score") +
  ggtitle("Results by model")

# Boxplot results per model and smoothing tecnique
ggplot(results, aes(x=SmoothingMethod, y=value)) +
  facet_wrap(~Model) +
  geom_boxplot() +
  xlab("Smoothing Method") +
  ylab("Score") +
  ggtitle("Results by model and smoothing method")







