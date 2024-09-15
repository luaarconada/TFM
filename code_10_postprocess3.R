###################################
#                                 
#             OVERVIEW            
#                                 
###################################

# This code implements Platt Scaling. The intermediate results are imported from 
# the files saved in `code_08_postprocess1.R`. The code saves final results for 
# Platt scaling in `results`.



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
data_list <- c('data1','datacorr1')

# partitioning
num_folds <- 10
seed      <- 1

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
for (data in data_list) {
  
  # read data
  dtest_unscaled <- read.csv(file.path(data_path, 'prepared', paste0(data, '_orig_test.csv')))

  # factor encoding
  dtest_unscaled$target  = as.factor(ifelse(dtest_unscaled$target == 2, 'Parole', 'Noparole'))
  dtest_unscaled$race     = as.factor(ifelse(dtest_unscaled$race == 1,    'White',  'Racialized'))
  dtest_unscaled$rehab.Not.participated   <- as.factor(ifelse(dtest_unscaled$rehab.Not.participated == 1,    'Norehab',  'Rehab'))
  dtest_unscaled$rehab.Participated   <- as.factor(ifelse(dtest_unscaled$rehab.Participated == 1,    'Rehab',  'Norehab'))


  # modeling
  #for (fold in seq(0, num_folds - 1)) {
    
    # feedback
    print('----------------------------------------')
    print(paste0('FOLD: ', fold))
    print('----------------------------------------')
    
    # import data
    dval_unscaled <- read.csv(file.path(data_path, 'prepared', paste0(data, '_orig_', fold, '_valid.csv')))
    dval_unscaled <- subset(dval_unscaled, select = c(race, target))
    
    # factor encoding
    dval_unscaled$target  = as.factor(ifelse(dval_unscaled$target == 2, 'Parole', 'Noparole'))
    dval_unscaled$race     = as.factor(ifelse(dval_unscaled$race == 1,    'White',  'Racialized'))
    #dval_unscaled$rehab.Not.participated   <- as.factor(ifelse(dval_unscaled$rehab.Not.participated == 1,    'Norehab',  'Rehab'))
    #dval_unscaled$rehab.Participated   <- as.factor(ifelse(dval_unscaled$rehab.Participated == 1,    'Rehab',  'Norehab'))
    
    # import preds
    dval_training_results  <- read.csv(file.path(res_path,'in2post1', paste0(data, '_', fold, '_AD_POST_training_results_dval.csv')))
    dtest_training_results <- read.csv(file.path(res_path, 'in2post1', paste0(data, '_', fold, '_AD_POST_training_results_dtest.csv')))
    
    
    # ---- PLATT SCALING PER GROUP ----
    
    # reload girds
    source(file.path(func_path, '98_param_grids.R'))
    
    # loop through sensitive groups
    platt_scores_list <- list()
    platt_valid_scores_list <- list()
    for (i in c('Racialized', 'White')) {
      
      # subset data
      dval_target  <- dval_unscaled$target[dval_unscaled$race    == i]
      dval_scores  <- dval_training_results[dval_unscaled$race   == i,]
      dtest_scores <- dtest_training_results[dtest_unscaled$race == i,]
      dtest_subset <- dtest_unscaled[dtest_unscaled$race         == i,]
      dval_subset  <- dval_unscaled[dval_unscaled$race           == i,]
      platt_scores       <- NULL
      platt_valid_scores <- NULL
      
      # perform scaling
      for (m in model.names) {
        
        # train logistic model with Yval ~ Y^val --> model_val
        dataframe_valid   <- data.frame(x = dval_scores[, paste0(m, '_scores')], y = dval_target)
        dataframe_valid$y <- ifelse(dataframe_valid$y == 'Parole', 1, 0)
        model_val         <- glm(y~x, data = dataframe_valid, family = binomial)
        
        # predict scores
        valid_scores       <- predict(model_val, newdata = dataframe_valid, type = 'response')
        platt_valid_scores <- cbind(platt_valid_scores, valid_scores)
        
        # use model_val to predict ytest
        dataframe_test <- data.frame(x = dtest_scores[, paste0(m, '_scores')])
        test_scores    <- predict(model_val, newdata = dataframe_test, type = 'response')
        platt_scores   <- cbind(platt_scores, test_scores)
      }
      colnames(platt_scores)       <- model.names
      colnames(platt_valid_scores) <- model.names
      platt_scores_list[[i]] <- cbind(platt_scores, dtest_subset)
      platt_valid_scores_list[[i]] <- cbind(platt_valid_scores, dval_subset)
    }
    platt_results       <- do.call(rbind, platt_scores_list)
    platt_valid_results <- do.call(rbind, platt_valid_scores_list)
    
    #----- THRESHOLDING ----
    
    # find optimal cutoff based on validation set
    for (m in model.names){
      pred <- platt_valid_results[, m]
      EMP  <- empCreditScoring(scores = pred, classes = platt_valid_results$target)
      assign(paste0('cutoff.', m), EMP$EMPCfrac)
    }
    
    
    #---- TESTING ----
    
    # assess test results
    test_results <- NULL
    for (m in model.names) {
      
      # load preds and scores
      pred         <- platt_results[, m]
      cutoff       <- quantile(pred, get(paste0('cutoff.', m)))
      cutoff_label <- sapply(pred, function(x) ifelse(x > cutoff, 'Parole', 'Noparole'))
      
      # evaluation
      res <- evaluate(class_preds = cutoff_label, 
                      score_preds = pred,
                      targets     = platt_results$target, 
                      healths     = platt_results$sentence, 
                      race         = platt_results$race,
                      r           = 0.753)
      test_results <- cbind(test_results, res)
    }
    
    # save results
    colnames(test_results) <- c(model.names)
    write.csv(test_results, file.path(res_path, 'in2post1post3final', paste0(data, '_', fold, '_PL_AD_results.csv')), row.names = TRUE)
  #}
}

