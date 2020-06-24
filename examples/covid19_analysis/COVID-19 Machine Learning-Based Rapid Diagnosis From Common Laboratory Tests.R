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
print(model.summary[1:10,])

lapply(as.character(model.summary$var[1:5]), plot.gbm, x=gbm.model)

plot.gbm(gbm.model, i.var = c(as.character(model.summary$var[1]), 'Patient.age.quantile'),  main="Rhinovirus.Enterovirus")
plot.gbm(gbm.model, i.var = c(as.character(model.summary$var[2]), 'Patient.age.quantile'),  main="Influenza.B")
plot.gbm(gbm.model, i.var = c(as.character(model.summary$var[3]), 'Patient.age.quantile'),  main="Leukocytes")
plot.gbm(gbm.model, i.var = c(as.character(model.summary$var[4]), 'Patient.age.quantile'),  main="Platelets") 
plot.gbm(gbm.model, i.var = c(as.character(model.summary$var[5]), 'Patient.age.quantile'),  main="Inf.A.H1N1.2009") 

library(pROC)
test.current.prediction <-predict(gbm.model, newdata = test, n.trees = 500,
                                   type="response")

x.roc<-roc(response=test$SARS.Cov.2.exam.result, predictor=test.current.prediction)

plot(x.roc, ylim=c(0,1),
     main=paste('AUC:',round(x.roc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

train.current.prediction <-predict(gbm.model, newdata = train, n.trees = 500,
                             type="response")
x.roc<-roc(response=train$SARS.Cov.2.exam.result, predictor=train.current.prediction)

cc <- coords(x.roc, seq(from = 0, to = 1, by = 0.05), ret=c("sensitivity", "specificity", "threshold"), transpose = FALSE)

library(ggplot2)
library(ggthemes)
mid<-median(cc$threshold)

ggplot(cc, aes(x=specificity, y=sensitivity,
               color=threshold, 
               fill=threshold)) + geom_point(size = 5) + geom_line() +
  theme_bw() +
  scale_color_gradient2(midpoint=mid, low="blue", mid="white", high="red", space ="Lab" ) +
  scale_fill_gradient2(midpoint=mid, low="blue", mid="white", high="red", space ="Lab" )

library(pROC)
train.current.prediction <-predict(gbm.model, newdata = train, n.trees = 500,
                             type="response")
                             

best.th<-coords(roc=x.roc, x=1, input="sensitivity", transpose = FALSE)$threshold
print(paste0("Optimal threshold = ", best.th))

oos.current.prediction <-predict(gbm.model, newdata = test, n.trees = 500,
                                   type="response")

print(paste0("Pct patients predicted as infected = ", sum(oos.current.prediction > best.th) / length(oos.current.prediction)))

oos.x.roc<-roc(test$SARS.Cov.2.exam.result, predictor=oos.current.prediction)

BinModelPerformance(oos.current.prediction, best.th, test$SARS.Cov.2.exam.result)

oos.current.prediction <-predict(gbm.model, newdata = test, n.trees = 500,
                                 type="response")


#obtain optimum threshold
best.th<-coords(x.roc, "best", ret="threshold", transpose = FALSE, 
                best.method="youden")$threshold
print(paste0("Optimal threshold = ", best.th))

print(paste0("Pct patients predicted as infected = ", 
             sum(oos.current.prediction > best.th) / length(oos.current.prediction)))

BinModelPerformance(oos.current.prediction, best.th,  test$SARS.Cov.2.exam.result)

oos.x.roc<-roc(test$SARS.Cov.2.exam.result, predictor=oos.current.prediction)

# OUT-OF-SAMPLE ROC
plot(oos.x.roc, ylim=c(0,1), print.thres="best", print.thres.best.method="youden",
     main=paste('AUC:',round(oos.x.roc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

library(pROC)
library(gbm)
prevalence <-0.1

summ<-list()
num.positive.pct<-num.positive<-auc.summ<-vector()
cost.min<-1
cost.max<-200
# to do Vectorize :( 
for(cost in seq(from = cost.min, to = cost.max, by = 1)){

#obtain optimum threshold
best.th<-coords(x.roc, "best", ret="threshold", transpose = FALSE, 
                best.method="youden", best.weights=c(cost, prevalence))$threshold 


oos.current.prediction <-predict(gbm.model, newdata = test, n.trees = 500,
                                 type="response")

#calculates pct/number of patients labeled as positive
num.positive.pct[cost]<-sum(oos.current.prediction > best.th) / length(oos.current.prediction)
num.positive[cost]<-sum(oos.current.prediction > best.th)

summ[[cost]]<-BinModelPerformance(oos.current.prediction, best.th, test$SARS.Cov.2.exam.result)

oos.x.roc<-roc(test$SARS.Cov.2.exam.result, predictor=oos.current.prediction)

auc.summ[cost]<-auc(oos.x.roc)

}

df.summ <- data.frame(matrix(unlist(summ), nrow=length(summ), byrow=T))
df.summ$num.positive <- num.positive
df.summ$num.positive.pct <- num.positive.pct
df.summ$auc <- auc.summ
df.summ$cost <- cost.min:cost.max
names(df.summ)<-c(names(summ[[1]]), "num.positive",  "num.positive.pct", "auc","cost")

df.summ$ROC<-NULL

library(reshape)
mdata <- melt(df.summ, id=c("cost","num.positive", "num.positive.pct"))

library(ggplot2)
ggplot(mdata, aes(x = cost, y = value)) + 
  geom_line(aes(color = variable)) + 
  theme_bw()

ggplot(mdata, aes(x = num.positive.pct, y = value)) + 
  geom_line(aes(color = variable)) + 
  theme_bw()
