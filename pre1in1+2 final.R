###################################
#                                 
#             OVERVIEW            
#                                 
###################################

# This code performs meta-parameter tuning of three in-processors: 
# - Prejudice Remover
# - Meta-Fair Algorithm
# - Adversarial Debiasing
#
# The code compares the EMP of these in-processors on validation folds. 
# The intermediate results are imported from the files exported in `code_05_inprocess1.ipynb`
# and `code_06_inprocess2.ipynb`. The code saves final results for in-processors
# in `results`.



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
#data_list <- paste0('data', 1:3)
data_list <- 'datacorr1'
# partitioning
num_folds <- 10
seed      <- 1

# adversary loss weight
#all_ad_adversary_loss_weight <- c(0.1, 0.01, 0.001)
all_ad_adversary_loss_weight <- 0.1
all_lambda <- c('0.5','0.6','0.7','0.8','0.9','1.0')

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
source(file.path(func_path, '99_compute_profit.R'))

# Data loop
  for (data in data_list) {
    # Modeling
    for (fold in seq(0, num_folds - 1)) {
      
      # Feedback
      print('----------------------------------------')
      print(paste0('FOLD: ', fold))
      print('----------------------------------------')
      
      # Read data
      dtest          <- read.csv(file.path(data_path, 'prepared/', paste0(data, '_scaled_', fold, '_test.csv')))
      dval           <- read.csv(file.path(data_path, 'prepared/', paste0(data, '_scaled_', fold, '_valid.csv')))
      dtest_unscaled <- read.csv(file.path(data_path, 'prepared/', paste0(data, '_orig_test.csv')))
      
      # Factor encoding
      dval$target            <- as.factor(ifelse(dval$target  == 2,          'Parole', 'Noparole'))
      dtest$target           <- as.factor(ifelse(dtest$target == 2,          'Parole', 'Noparole'))
      dtest_unscaled$target  <- as.factor(ifelse(dtest_unscaled$target == 2, 'Parole', 'Noparole'))
      dval$race               <- as.factor(ifelse(dval$race  == 1,             'White',  'Racialized'))
      dtest$race              <- as.factor(ifelse(dtest$race == 1,             'White',  'Racialized'))
      dtest_unscaled$race     <- as.factor(ifelse(dtest_unscaled$race == 1,    'White',  'Racialized'))
      dval$rehab.Not.participated     <- as.factor(ifelse(dval$rehab.Not.participated   == 1,   'Norehab',  'Rehab'))
      dtest$rehab.Not.participated    <- as.factor(ifelse(dtest$rehab.Not.participated  == 1,   'Norehab',  'Rehab'))
      dtest_unscaled$rehab.Not.participated   <- as.factor(ifelse(dtest_unscaled$rehab.Not.participated == 1,    'Norehab',  'Rehab'))
      dval$rehab.Participated     <- as.factor(ifelse(dval$rehab.Participated   == 1,   'Rehab',  'Norehab'))
      dtest$rehab.Participated    <- as.factor(ifelse(dtest$rehab.Participated  == 1,   'Rehab',  'Norehab'))
      dtest_unscaled$rehab.Participated   <- as.factor(ifelse(dtest_unscaled$rehab.Participated == 1,    'Rehab',  'Norehab'))
      
      #-------------------------- PREJUDICE REMOVER ----------------------------------
      
      # Feedback
      print('- PREJUDICE REMOVER...')
      
      # Load predictions for DI_lambda
      for (lambda in all_lambda) {
        dval_pred  <- read.csv(file.path(res_path, 'pre1in1', paste0(data, '_', fold, '_PR_predictions_valid_DI_', lambda, '.csv')))
        dtest_pred <- read.csv(file.path(res_path, 'pre1in1', paste0(data, '_', fold, '_PR_predictions_test_DI_', lambda, '.csv')))
        
        #---- THRESHOLDING ----
        
        # Find optimal cutoff based on validation set
        empVals <- NULL
        for (col in 1:ncol(dval_pred)) {
          empVal  <- empCreditScoring(dval_pred[, col], dval$target)
          empVals <- unlist(c(empVals, empVal['EMPC']))
        }
        bestPrediction <- dval_pred[, which(empVals == max(empVals))[1]]
        best_eta       <- colnames(dval_pred)[which(empVals == max(empVals))[1]]
        
        # Define cutoff
        EMP    <- empCreditScoring(scores = bestPrediction, classes = dval$target)
        cutoff <- EMP$EMPCfrac
        
        #---- TESTING ----
        
        # Extract preds and scores
        pred         <- dtest_pred[, best_eta]
        cutoff       <- quantile(pred, cutoff)
        cutoff_label <- sapply(pred, function(x) ifelse(x > cutoff, 'Parole', 'Noparole'))
        
        # Evaluation
        res <- evaluate(class_preds = cutoff_label, 
                        score_preds = pred,
                        targets     = dtest_unscaled$target,
                        healths     = dtest_unscaled$sentence,
                        race        = dtest_unscaled$race,
                        r           = 0.753)
        
        # Save results
        write.csv(res, file.path(res_path, 'pre1in1final', paste0(data, '_', fold, '_PR_results_DI_', lambda, '.csv')), row.names = T)
      }
      
      # Load predictions for RW
      dval_pred  <- read.csv(file.path(res_path, 'pre1in1', paste0(data, '_', fold, '_PR_predictions_valid_RW.csv')))
      dtest_pred <- read.csv(file.path(res_path, 'pre1in1', paste0(data, '_', fold, '_PR_predictions_test_RW.csv')))
      
      #---- THRESHOLDING ----
      
      # Find optimal cutoff based on validation set
      empVals <- NULL
      for (col in 1:ncol(dval_pred)) {
        empVal  <- empCreditScoring(dval_pred[, col], dval$target)
        empVals <- unlist(c(empVals, empVal['EMPC']))
      }
      bestPrediction <- dval_pred[, which(empVals == max(empVals))[1]]
      best_eta       <- colnames(dval_pred)[which(empVals == max(empVals))[1]]
      
      # Define cutoff
      EMP    <- empCreditScoring(scores = bestPrediction, classes = dval$target)
      cutoff <- EMP$EMPCfrac
      
      #---- TESTING ----
      
      # Extract preds and scores
      pred         <- dtest_pred[, best_eta]
      cutoff       <- quantile(pred, cutoff)
      cutoff_label <- sapply(pred, function(x) ifelse(x > cutoff, 'Parole', 'Noparole'))
      
      # Evaluation
      res <- evaluate(class_preds = cutoff_label, 
                      score_preds = pred,
                      targets     = dtest_unscaled$target,
                      healths     = dtest_unscaled$sentence,
                      race        = dtest_unscaled$race,
                      r           = 0.753)
      
      # Save results
      write.csv(res, file.path(res_path, 'pre1in1final', paste0(data, '_', fold, '_PR_results_RW.csv')), row.names = T)
      
      #-------------------------- ADVERSARIAL DEBIASING ------------------------------
      
      # Feedback
      #print('- ADVERSARIAL DEBIASING...')
      
      #---- TUNING ----
      
      # Placeholder
      #emp_dval <- NULL
      
      # Tune meta-parameter
      #for (lambda in all_lambda) {
        
        # Import predictions
        #dval_pred  <- read.csv(file.path(res_path, 'pre1in2', paste0(data, '_', fold, '_AD_DI_', lambda, '_predictions_valid.csv')))
        
        # Write EMP
        #EMP <- empCreditScoring(dval_pred[, 'scores'], 2 - dval_pred[, 'targets'])$EMPC
        #emp_dval <- rbind(emp_dval, c(as.numeric(lambda), EMP))
      #}
      
      # Format results
      #emp_dval <- data.frame(emp_dval)
      #colnames(emp_dval) <- c('lambda', 'EMP')
      
      # Find optimal lambda
      #optimal_lambda <- emp_dval$lambda[which.max(emp_dval$EMP)]
      
      # Import relevant predictions
      #dval_pred  <- read.csv(file.path(res_path, 'pre1in2', paste0(data, '_', fold, '_AD_DI_', optimal_lambda, '_predictions_valid.csv')))
      #dtest_pred <- read.csv(file.path(res_path, 'pre1in2', paste0(data, '_', fold, '_AD_DI_', optimal_lambda, '_predictions_test.csv')))
      
      # Load predictions for RW
      #dval_pred  <- read.csv(file.path(res_path, 'pre1in2', paste0(data, '_', fold, '_AD_RW_predictions_valid.csv')))
      #dtest_pred <- read.csv(file.path(res_path, 'pre1in2', paste0(data, '_', fold, '_AD_RW_predictions_test.csv')))
      
      #---- THRESHOLDING ----
      
      # Find optimal cutoff
      #EMP    <- empCreditScoring(dval_pred[, 'scores'], 2 - dval_pred[, 'targets'])
      #cutoff <- EMP$EMPCfrac
      
      #---- TESTING ----
      
      # Extract preds and scores
      #pred         <- dtest_pred[, 'scores']
      #cutoff       <- quantile(pred, cutoff)
      #cutoff_label <- sapply(pred, function(x) ifelse(x > cutoff, 'Parole', 'Noparole'))
      
      # Evaluation
      #res <- evaluate(class_preds = cutoff_label,
                      #score_preds = pred,
                      #targets     = dtest_unscaled$target,
                      #healths     = dtest_unscaled$sentence,
                      #race        = dtest_unscaled$race,
                      #r           = 0.753)
      
      # Save results
      #write.csv(res, file.path(res_path, 'pre1in2final', paste0(data, '_', fold, '_AD_results.csv')), row.names = T)
    }
  }
