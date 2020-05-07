# 00-Funcs.R
# current.prediction<-oos.current.prediction
# prediction.threshold <- best.th
# reference <- test$SARS.Cov.2.exam.result

BinModelPerformance <- function(current.prediction, prediction.threshold, reference){
  
  classes = c("positive", "negative")
  
  current.prediction.outcome <- as.factor(ifelse(current.prediction>prediction.threshold, "positive", "negative"))
  reference = as.factor(ifelse(reference, "positive", "negative"))
  
  current.prediction.outcome <- factor(current.prediction.outcome, levels =   classes)
  reference<- factor( reference, levels =   classes)
  
  # Confusion matrix
  print(caret::confusionMatrix(current.prediction.outcome, reference = reference, positive="positive")$table)
  
  x<-data.frame(obs=reference, pred=current.prediction.outcome)
  x$positive <- current.prediction
  x$negative <- 1 -  current.prediction
  #ROC      Sens      Spec
  pr2ClassSumm<-caret::twoClassSummary(x, lev=   classes )
 # print(pr2ClassSumm)
  
  model.summ <- c(pr2ClassSumm)

  return(model.summ) 
  
}


### Drop samples that do not have enough non-NA values
delete.na <- function(DF, n=0, is.row=TRUE) {
  if(is.row){
    DF[rowSums(!is.na(DF)) >= n,]
  }else{
    DF[,colSums(!is.na(DF)) >= n]
  }
}
