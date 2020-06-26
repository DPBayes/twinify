# Derived from code of Souza et al.
# https://github.com/souzatharsis/covid-19-ML-Lab-Test/blob/master/src/COVID-19%20Machine%20Learning-Based%20Rapid%20Diagnosis%20From%20Common%20Laboratory%20Tests.ipynb

# load aux functions
source("../00-Funcs.R")

#load input dataset
library("readxl");
data <- as.data.frame(read_excel("../dataset.xlsx"), stringAsFactors=F)

# make variable names syntactically valid
names(data) <- make.names(names(data))
data$Patient.ID <- NULL

# Replace column values that should be empty for NA
data[data=='Não Realizado'] <- NA
data[data=='not_done'] <- NA
data[data=='<1000'] <- 500

data$Urine...Leukocytes <- as.integer(data$Urine...Leukocytes)
data$Urine...pH <- as.integer(data$Urine...pH)

# convert string values to factors
ind <- sapply(data, is.character)
data[ind] <- lapply(data[ind], factor)

data$Lipase.dosage <- as.factor(data$Lipase.dosage)

outcome.var<-"SARS.Cov.2.exam.result"
data[, outcome.var] <- as.integer(data[, outcome.var]) - 1

#countNA<-function(x){sum(is.na(x))}
#x<-apply(data, 2, countNA)/nrow(data)

data.saved<-data
data.size<-nrow(data)
not.na.pct <- 0.05
data <- delete.na(data, n = data.size * not.na.pct, is.row = FALSE)


data.pos <- data[data$SARS.Cov.2.exam.result==1,]
data.neg <-  data[data$SARS.Cov.2.exam.result==0,]

### delete poor samples
min.non.na.vars <- 10
data.neg <- delete.na(data.neg, n = min.non.na.vars)

data <- rbind(data.pos, data.neg)

library(caret)
set.seed(18101987)
SPLIT.RATIO <- 2/3
train.index <- createDataPartition(data$SARS.Cov.2.exam.result, p = SPLIT.RATIO, list = FALSE)
train <- data[train.index,]
test <- data[-train.index,]

train.features <- setdiff(names(train), c(outcome.var, "Patient.ID"))
myformula = as.formula(paste0(outcome.var," ~ ", paste0(train.features, collapse="+")))

BAG.FRACTION <- 0.8
library(gbm)
gbm.model = gbm(myformula, data = train,
                n.trees = 500 ,
                bag.fraction = BAG.FRACTION,
                verbose=FALSE)

model.summary <- summary(gbm.model, plotit=FALSE)

## extract feature names
original_data <- as.data.frame(read_excel("../dataset.xlsx"), stringAsFactors=F)
all_names = as.vector(model.summary$var)
all_new_names <- rep(NA, length(all_names))
for (i in 1:length(all_names)){
  name <- names(original_data)[make.names(names(original_data))==all_names[i]]
  all_new_names[i] <- name
}
all_new_names <- c(names(original_data)[make.names(names(original_data)) == outcome.var], all_new_names)
write(as.vector(all_new_names), "covid19_features.txt")
