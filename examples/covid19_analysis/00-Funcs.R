# Due to Souza et al.
# excerpt from: https://github.com/souzatharsis/covid-19-ML-Lab-Test/blob/master/src/00-Funcs.R

### Drop samples that do not have enough non-NA values
delete.na <- function(DF, n=0, is.row=TRUE) {
  if(is.row){
    DF[rowSums(!is.na(DF)) >= n,]
  }else{
    DF[,colSums(!is.na(DF)) >= n]
  }
}
