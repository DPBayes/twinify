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

# convert string values to factors
original.ind <- sapply(original.data, is.character)
original.data[original.ind] <- lapply(original.data[original.ind], factor)

# encode outcome variable as 0/1
original.data[, outcome.var] <- as.integer(original.data[, outcome.var]) - 1

original.data.saved<-original.data
original.data.size<-nrow(original.data)

## delete features that have >95% missing values
not.na.pct <- 0.05
original.data <- delete.na(original.data, n = original.data.size * not.na.pct, is.row = FALSE)

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
library(pROC)
library(gbm)
set.seed(18101987)
SPLIT.RATIO <- 2/3
original.train.index <- createDataPartition(original.data$SARS.Cov.2.exam.result, p = SPLIT.RATIO, list = FALSE)
original.train <- original.data[original.train.index,]
original.test <- original.data[-original.train.index,]

original.train.features <- setdiff(names(original.train), c(outcome.var, "Patient.ID"))
myformula = as.formula(paste0(outcome.var," ~ ", paste0(original.train.features, collapse="+"))) ## formula for GBM model

BAG.FRACTION <- 0.8

## Baseline results from original data
original.gbm.model = gbm(myformula, data = original.train,
                         n.trees = 500 ,
                         bag.fraction = BAG.FRACTION,
                         verbose=FALSE)

# compute prediction
original.test.prediction <-predict(original.gbm.model, newdata = original.test, n.trees = 500,
                                   type="response")
# compute roc
original.roc<-roc(response=original.test$SARS.Cov.2.exam.result, predictor=original.test.prediction)


## Optimal threshold with priority on sensitivity, 2nd plot
original.train.prediction <-predict(original.gbm.model, newdata = original.train, n.trees = 500,
                                    type="response")
original.train.roc <- roc(response=original.train$SARS.Cov.2.exam.result, predictor=original.train.prediction)

orig.best.th.sens <- coords(roc=original.train.roc, x=1, input="sensitivity", transpose = FALSE)$threshold
orig.res.sens.opt <- BinModelPerformance(original.test.prediction, orig.best.th.sens, original.test$SARS.Cov.2.exam.result)

## Optimal threshold with priority on balance, 3nd plot
orig.best.th.bal <- coords(original.train.roc, "best", ret="threshold", transpose = FALSE, best.method="youden")$threshold
orig.res.bal.opt <- BinModelPerformance(original.test.prediction, orig.best.th.bal,  original.test$SARS.Cov.2.exam.result)


###################################################
## load synthetic data
epsilons <- c(2.0)
nruns <- 10

# data frames for storing the results
basic.auc.df <- data.frame(matrix(vector(), nruns, length(epsilons), dimnames = list(c(), factor(epsilons)) ))

sens.auc.df <- data.frame(matrix(vector(), nruns, length(epsilons), dimnames = list(c(), factor(epsilons)) ))
sens.Sens.df <- data.frame(matrix(vector(), nruns, length(epsilons), dimnames = list(c(), factor(epsilons)) ))
sens.Spec.df <- data.frame(matrix(vector(), nruns, length(epsilons), dimnames = list(c(), factor(epsilons)) ))

bal.auc.df <- data.frame(matrix(vector(), nruns, length(epsilons), dimnames = list(c(), factor(epsilons)) ))
bal.Sens.df <- data.frame(matrix(vector(), nruns, length(epsilons), dimnames = list(c(), factor(epsilons)) ))
bal.Spec.df <- data.frame(matrix(vector(), nruns, length(epsilons), dimnames = list(c(), factor(epsilons)) ))

eps.iter <- 0
for (epsilon in epsilons) {
  eps.iter <- eps.iter + 1
  seed.iter <- 0
  for (seed in c(0:(nruns-1))) {
    seed.iter <- seed.iter + 1
    synthetic.data.fname <- sprintf("./full_model_nonprivate/syn_data_seed%d_eps%.1f.csv", seed, epsilon)
    synthetic.data <- as.data.frame(read.csv(synthetic.data.fname, na.strings = ""), stringAsFactors=F)
    synthetic.data$X <- NULL
    
    # convert string values to factors
    synthetic.ind <- sapply(synthetic.data, is.character)
    synthetic.data[synthetic.ind] <- lapply(synthetic.data[synthetic.ind], factor)
    
    # encode outcome variable as 0/1
    synthetic.data[, outcome.var] <- as.integer(synthetic.data[, outcome.var]) - 1
    
    synthetic.data.saved<-synthetic.data
    synthetic.data.size<-nrow(synthetic.data)
    
    ## delete features that have >95% missing values
    synthetic.data <- delete.na(synthetic.data, n = synthetic.data.size * not.na.pct, is.row = FALSE)
    
    ## separate data into positive and negative based on outcome
    synthetic.data.pos <- synthetic.data[synthetic.data$SARS.Cov.2.exam.result==1,]
    synthetic.data.neg <-  synthetic.data[synthetic.data$SARS.Cov.2.exam.result==0,]
    
    ### delete poor samples
    min.non.na.vars <- 10
    synthetic.data.neg <- delete.na(synthetic.data.neg, n = min.non.na.vars)
    
    synthetic.data <- rbind(synthetic.data.pos, synthetic.data.neg)
    
    
    ####### now learn predictive model with synthetic data
    synthetic.gbm.model = gbm(myformula, data = synthetic.data,
                              n.trees = 500 ,
                              bag.fraction = BAG.FRACTION,
                              verbose=FALSE)
    
    ## Basic prediction task for 1st bar plot
    # compute prediction
    synthetic.test.prediction <-predict(synthetic.gbm.model, newdata = original.test, n.trees = 500,
                                   type="response")
    #compute roc
    synthetic.roc<-roc(response=original.test$SARS.Cov.2.exam.result, predictor=synthetic.test.prediction)
    
    basic.auc.df[seed.iter, eps.iter] <- synthetic.roc$auc
    
    ## Optimal threshold with priority on sensitivity, 2nd plot
    synthetic.train.prediction <-predict(synthetic.gbm.model, newdata = synthetic.data, n.trees = 500,
                                       type="response")
    synthetic.train.roc <- roc(response=synthetic.data$SARS.Cov.2.exam.result, predictor=synthetic.train.prediction)
    
    best.th.sens <- coords(roc=synthetic.train.roc, x=1, input="sensitivity", transpose = FALSE)$threshold
    
    res.sens.opt <- BinModelPerformance(synthetic.test.prediction, best.th.sens, original.test$SARS.Cov.2.exam.result)
    sens.auc.df[seed.iter, eps.iter] <- res.sens.opt[1]
    sens.Sens.df[seed.iter, eps.iter] <- res.sens.opt[2]
    sens.Spec.df[seed.iter, eps.iter] <- res.sens.opt[3]
    
    ## Optimal threshold with priority on balance, 3nd plot
    synthetic.train.prediction <-predict(synthetic.gbm.model, newdata = synthetic.data, n.trees = 500,
                                         type="response")
    synthetic.train.roc <- roc(response=synthetic.data$SARS.Cov.2.exam.result, predictor=synthetic.train.prediction)
    best.th.bal <- coords(synthetic.train.roc, "best", ret="threshold", transpose = FALSE, best.method="youden")$threshold
    res.bal.opt <- BinModelPerformance(synthetic.test.prediction, best.th.bal,  original.test$SARS.Cov.2.exam.result)
    bal.auc.df[seed.iter, eps.iter] <- res.bal.opt[1]
    bal.Sens.df[seed.iter, eps.iter] <- res.bal.opt[2]
    bal.Spec.df[seed.iter, eps.iter] <- res.bal.opt[3]
  }
}
# save results
write.csv(basic.auc.df, file="./r_outputs/basic_auc_full_model_nonprivate.csv")

write.csv(sens.auc.df, file="./r_outputs/sens_auc_full_model_nonprivate.csv")
write.csv(sens.Sens.df, file="./r_outputs/sens_sens_full_model_nonprivate.csv")
write.csv(sens.Spec.df, file="./r_outputs/sens_spec_full_model_nonprivate.csv")

write.csv(bal.auc.df, file="./r_outputs/bal_auc_full_model_nonprivate.csv")
write.csv(bal.Sens.df, file="./r_outputs/bal_sens_full_model_nonprivate.csv")
write.csv(bal.Spec.df, file="./r_outputs/bal_spec_full_model_nonprivate.csv")

