library("googlesheets")
library("dplyr")
library("reshape2")
library("ggplot2")
library("sqldf")
library("ggthemes")
library("Cairo")

###########################################
##      Fetch data and cleanup
###########################################


# Fetch sentences and survey results from Google Docs spreadsheet
survey_sheet <- gs_key("1u6kJ8Z4YzhaGiOMFXGrPvqu8Cm6_yeWchulllF4k-sg")
sentences <- survey_sheet %>% gs_read(ws = "Sentences")

raw_results <- survey_sheet %>% gs_read(ws = "Form Responses 1")
raw_results <- melt(raw_results)
colnames(raw_results)[2] <- "Sentence"
raw_results <- raw_results[,-1] # remove the timestamp
raw_results$Sentence <- gsub("\\.", " ", raw_results$Sentence) # "a.b.c" -> "a b c"


# combine the sentences and survey results into one data frame
results <- inner_join(raw_results, sentences)

# Factorize for proper ordering on graphs

resultsForModels <- filter(results, Model != "Perfect" & Model != "Random")
resultsForPerfectAndBaseline <- filter(results, Model == "Perfect" | Model == "Random")

results$SmoothingMethod <- factor(results$SmoothingMethod, 
                                  levels = c("Random", "MLEProbDist", "LaplaceProbDist", "ELEProbDist", "SimpleGoodTuringProbDist", "Perfect"),
                                  labels = c("Random", "MLE", "Laplace", "ELE", "Good Turing", "Perfect"),
                                  ordered = TRUE)
results$Model <- factor(results$Model, 
                                  levels = c("Perfect", "SmoothOperator", "InferredGrammar", "Random"),
                                  labels = c("Human", "Semantic Model", "Inferred Grammar", "Baseline"),
                                  ordered = TRUE)


resultsForModels$SmoothingMethod <- factor(resultsForModels$SmoothingMethod, 
                                  levels = c("MLEProbDist", "LaplaceProbDist", "ELEProbDist", "SimpleGoodTuringProbDist"),
                                  labels = c("MLE", "Laplace", "ELE", "Good Turing"),
                                  ordered = TRUE)
resultsForModels$Model <- factor(resultsForModels$Model, 
                        levels = c("SmoothOperator", "InferredGrammar"),
                        labels = c("Semantic Model", "Inferred Grammar"),
                        ordered = TRUE)




# average response per model and smoothing method
Mode <- function(x) {
  ux <- unique(x[!is.na(x)])
  ux[which.max(tabulate(match(x, ux)))]
}

meansByModelAndMethod <- resultsForModels[2:4] %>% 
                         group_by(Model, SmoothingMethod) %>% 
                         select(value) %>%
                         summarise(means = mean(value, na.rm = T), 
                                   median = median(value, na.rm = T),
                                   mode = Mode(value))
melt(meansByModelAndMethod)
ggplot(melt(meansByModelAndMethod), aes(x = SmoothingMethod, y=value)) + geom_bar()

meansByModel <- results[2:4] %>% 
                group_by(Model) %>% 
                select(value) %>%
                summarise(means = mean(value, na.rm = T),
                          median = median(value, na.rm = T),
                          mode = Mode(value))



###########################################
##      Graphical results
###########################################
resultsFolder <- "/Users/kristo/Google Drive/KTH/Courses/Artificial Intelligence/AI-FinalProject/report/pictures/results/"

# Histogram results per model and smooothing method
png(filename=paste(resultsFolder,  "histogram_resultsByModelAndSmootingMethod.png", sep=""), 
    type="cairo",
    units="px", 
    width=800, 
    height=600, 
    pointsize=12, 
    res=96)

ggplot(resultsForModels, aes(x=value)) +
  facet_grid(Model~SmoothingMethod) +
  scale_x_discrete(limits=c(1,2,3,4,5)) +
  geom_histogram(aes(y=..density..), alpha=I(0.5), color="white", fill="black") +
  geom_vline(data=meansByModelAndMethod, aes(xintercept=means), colour="black", linetype="dashed", size=1) +
  xlab("Score") +
  #ggtitle("Results by model and smoothing method") +
  theme_bw(base_family = "Roboto Regular", base_size = 16)

dev.off()

# Histogram results per model
png(filename=paste(resultsFolder, "histogram_resultsByModel.png", sep=""), 
    type="cairo",
    units="px", 
    width=600, 
    height=600, 
    pointsize=12, 
    res=96)

ggplot(results, aes(x=value)) +
  facet_wrap(~Model) +
  scale_x_discrete(limits=c(1,2,3,4,5)) +
  geom_histogram(aes(y=..density..), alpha=I(0.5), color="white", fill="black") +
  geom_vline(data=meansByModel, aes(xintercept=means), linetype="dashed", size=1, colour="black") +
  xlab("Score") +
  #ggtitle("Results by model") +
  theme_bw(base_family = "Roboto Regular", base_size = 16)
  
dev.off()


# Boxplot results per model and smoothing tecnique
  png(filename=paste(resultsFolder, "boxplot_resultsByModelAndSmoothing.png", sep=""), 
      type="cairo",
      units="px", 
      width=900, 
      height=700, 
      pointsize=12, 
      res=121)
  
  ggplot(resultsForModels, aes(x=SmoothingMethod, y=value)) +
    facet_grid(.~Model) +
    geom_boxplot() +
    stat_summary(fun.y=mean, geom="point", shape=5, size=4) +
    xlab("Smoothing Method") +
    ylab("Score") +
    #ggtitle("Performance of model and smoothing method")+
    theme_bw(base_family = "Roboto Regular", base_size = 20) +
    theme(axis.text.x = element_text(size = 16, angle = 90, vjust = 1, hjust=1)) +
    theme(plot.title = element_text(size = 16))
  
  dev.off()


png(filename=paste(resultsFolder, "boxplot_resultsForPerfectAndRandom.png", sep=""), 
    type="cairo",
    units="px", 
    width=400, 
    height=600, 
    pointsize=12, 
    res=96)

ggplot(resultsForPerfectAndBaseline, aes(x=Model, y=value)) +
  #facet_wrap(~Model) +
  geom_boxplot() +
  stat_summary(fun.y=mean, geom="point", shape=5, size=4) +
  xlab("Smoothing Method") +
  ylab("Score") +
  #ggtitle("Results for...") +
  theme_bw(base_family = "Roboto Regular")

dev.off()

# Boxplot results per model

results$Model <- factor(results$Model, 
                        #levels = c("Perfect", "SmoothOperator", "InferredGrammar", "Random"),
                        labels = c("Human", "Semantic Mdl", "Inferred Gr", "Baseline"),
                        ordered = TRUE)

png(filename=paste(resultsFolder, "boxplot_resultsByModel.png", sep=""), 
    type="cairo",
    units="px", 
    width=500, 
    height=700, 
    pointsize=16, 
    res=121)

ggplot(results, aes(x=Model, y=value)) +
  geom_boxplot() +
  stat_summary(fun.y=mean, geom="point", shape=5, size=4) +
  xlab("Model")  + 
  ylab("Score") +
  #ggtitle("Performance of models") +
  theme_bw(base_family = "Roboto Regular", base_size = 24) +
  theme(axis.text.x = element_text(size = 20, angle = 45, vjust = 1, hjust=1)) +
  theme(plot.title = element_text(size = 18))
#theme(axis.text.x = element_text(size = 20, vjust = 1, hjust=1)) +
#coord_flip()

dev.off()




