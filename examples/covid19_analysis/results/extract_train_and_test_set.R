print("splitting train and test set for gbm analysis")

# load aux functions
source("../00-Funcs.R")

## set target variable name
outcome.var<-"SARS.Cov.2.exam.result"

## load original dataset
library("readxl");
original.data <- as.data.frame(read_excel("../dataset.xlsx"), stringAsFactors=F)

# make variable names syntactically valid
names(original.data) <- make.names(names(original.data))
original.data$Patient.ID <- NULL

## Pick features that match synthetic data
#original.data <- original.data[, features]

# Replace column values that should be empty for NA
original.data[original.data=='Não Realizado'] <- NA
original.data[original.data=='not_done'] <- NA
original.data[original.data=='<1000'] <- 500

original.data.saved<-original.data
original.data.size<-nrow(original.data)

## delete features that have >95% missing values
not.na.pct <- 0.05
original.data <- delete.na(original.data, n = original.data.size * not.na.pct, is.row = FALSE)

# convert string values to factors
original.ind <- sapply(original.data, is.character)
original.data[original.ind] <- lapply(original.data[original.ind], factor)

# encode outcome variable as 0/1
original.data[, outcome.var] <- as.integer(original.data[, outcome.var]) - 1

## separate data into positive and negative based on outcome
original.data.pos <- original.data[original.data$SARS.Cov.2.exam.result==1,]
original.data.neg <-  original.data[original.data$SARS.Cov.2.exam.result==0,]

### delete poor samples
min.non.na.vars <- 10
original.data.neg <- delete.na(original.data.neg, n = min.non.na.vars)

original.data <- rbind(original.data.pos, original.data.neg)


#####################################################
####### Learn predictive model with original data
library(caret)
set.seed(18101987)
SPLIT.RATIO <- 2/3
original.train.index <- createDataPartition(original.data$SARS.Cov.2.exam.result, p = SPLIT.RATIO, list = FALSE)
original.train <- original.data[original.train.index,]
write.csv(original.train, "original_train_gbm.csv", row.names = FALSE)
original.test <- original.data[-original.train.index,]
write.csv(original.test, "original_test_gbm.csv", row.names = FALSE)
