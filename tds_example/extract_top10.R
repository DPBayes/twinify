# load aux functions
source("./00-Funcs.R")

#load input dataset
library("readxl");
data <- as.data.frame(read_excel("dataset.xlsx"), stringAsFactors=F)

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

countNA<-function(x){sum(is.na(x))}
x<-apply(data, 2, countNA)/nrow(data)
plot(density(x/nrow(data)), main="Most variables have a high percentage of missing values", xlab="Percentage of NA's per variable")

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

print(setdiff(names(data.saved), names(data)))

print(names(data))

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

model.summary<-summary(gbm.model, cBars=10)
top_names = as.vector(model.summary[1:10,]$var)
print(model.summary[1:10,])
original_data <- as.data.frame(read_excel("dataset.xlsx"), stringAsFactors=F)
new_names <- rep(NA, 10)
for (i in 1:10){
  name <- names(original_data)[make.names(names(original_data))==top_names[i]]
  new_names[i] <- name
}
print(new_names)
write(as.vector(new_names), "tds_top10_features.txt")
