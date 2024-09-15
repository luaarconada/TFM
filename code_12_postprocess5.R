###################################
#                                 
#             OVERVIEW            
#                                 
###################################

# This code processes intermediate results of Reject Option Classification imported 
# from the files saved in `code_11_postprocess4.ipynb`. The code saves final results for 
# reject option classification in `results`.



###################################
#                                 
#             SETTINGS            
#                                 
###################################

# clearing the memory
rm(list = ls())

# installing pacman
if (require(pacman) == F) install.packages('pacman')
library(pacman)

# libraries
p_load(caret, doParallel, kernlab, randomForest, nnet, 
       xgboost, foreach, e1071, pROC, EMP)

# working directory
cd <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(dirname(cd))



###################################
#                                 
#           PARAMETERS            
#                                 
###################################

# paths
source(file.path(cd, 'code_00_working_paths.R'))

# data 
data_list <- c('data1', 'datacorr1')

# partitioning
num_folds <- 10
seed      <- 1

# metric bound
roc_bounds <- c(0.1,0.2,0.3)
# options
set.seed(seed)
options(scipen = 10)



##################################
#                                
#          MODELING      
#                                
##################################

# helper functions
source(file.path(func_path, '94_evaluate.R'))
source(file.path(func_path, '95_fairness_metrics.R'))
source(file.path(func_path, '96_emp_summary.R'))
source(file.path(func_path, '97_caret_settings.R'))
source(file.path(func_path, '98_param_grids.R'))
source(file.path(func_path, '99_compute_profit.R'))


fold <- 0

# data loop
for (data in data_list){

  # read data
  dtest_unscaled <- read.csv(file.path(data_path, 'prepared', paste0(data, '_orig_test.csv')))

  # factor encoding
  dtest_unscaled$target <- as.factor(ifelse(dtest_unscaled$target == 1, 'Parole', 'Noparole'))
  dtest_unscaled$race    <- as.factor(ifelse(dtest_unscaled$race == 1,    'White',  'Racialized'))

  # roc_bound loop
  for (roc_bound in roc_bounds) {
    # modeling
    #for (fold in seq(0, num_folds - 1)) {
  
      # feedback
      print('----------------------------------------')
      print(paste0('FOLD: ', fold))
      print('----------------------------------------')
 
      # load data and results
      POST_results <- read.csv(file.path(res_path, 'in2post1post4', paste0(data, '_', fold, '_ROC_', roc_bound, '_AD_predictions_test.csv')))
  
  
      #---- TESTING ----
  
      # reload grids
      source(file.path(func_path, '98_param_grids.R'))

      # placeholder
      test_results <- NULL
  
      # loop through base models
      for (m in model.names) {
    
        # load preds and scores
        cutoff_label <- POST_results[, paste0(m, '_fairLabels')]
        cutoff_label <- factor(as.character(cutoff_label), levels = c('Parole', 'Noparole'))
        scores       <- sapply(cutoff_label, function(x) ifelse(x == 'Parole', 1, 0))
    
        # evaluation
        res <- evaluate(class_preds = cutoff_label, 
                        score_preds = scores,
                        targets     = dtest_unscaled$target, 
                        healths     = dtest_unscaled$sentence, 
                        race        = dtest_unscaled$race,
                        r           = 0.753)
        test_results <- cbind(test_results, res)
      }  
  
      # save results
      colnames(test_results) <- c(model.names)
      write.csv(test_results, file.path(res_path, 'in2post1post4post5final', paste0(data, '_', fold, '_', roc_bound, '_AD_ROC_results.csv')), row.names = T)
    #}
  }
}
