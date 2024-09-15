###################################
#                                 
#             OVERVIEW            
#                                 
###################################

# This code implements a set of base classifiers using fair pre-processed data
# by two pre-processors: Reweighting and Disparate Impact Remover. Pre-processed  
# data are imported from the files saved in `code_01_preprocess1.ipynb`. The code
# saves intermediate results for Disparate Impact Remover and final results for 
# Reweighting in the corresponding subfolders in `results`.


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
data_list <- paste0('datacorr',1:10)

# partitioning
num_folds <- 10
seed      <- 1

# cores
cores <- 8

# repair level
all_di_repair_level <- c('0.6', '0.7', '0.8', '0.9', '1.0')
method <- "DI"

# options
set.seed(seed)
options(scipen = 10)

# parallel computing
nrOfCores <- cores
library(doParallel)
registerDoParallel(cores = nrOfCores)
message(paste('\n Registered number of cores:\n',nrOfCores,'\n'))


##################################
#                                
#          MODELING      
#                                
##################################

# helper functions
source(file.path(func_path, '94_evaluate.R'))
source(file.path(func_path, '95_fairness_metrics.R'))
source(file.path(func_path, '96_emp_summary.R'))
library(caret)
source(file.path(func_path, '97_caret_settings.R'))
source(file.path(func_path, '98_param_grids.R'))
source(file.path(func_path, '99_compute_profit.R'))


# dataset loop
for (data in data_list) {
  # modeling loop
  for (di_repair_level in all_di_repair_level) {
    for (fold in seq(0, 9)) {
      
      #---- PREPARATIONS ----
      
      # feedback
      print('----------------------------------------')
      print(paste0('METHOD: ', method, ' | FOLD: ', fold))
      print('----------------------------------------')      
      
      # read data
      if (method == 'DI') {
        dtest  <- read.csv(file.path(res_path, 'preprocess1', paste0(data, '_', fold, '_pre_test_',  method, '_', di_repair_level, '.csv')))
        dval   <- read.csv(file.path(res_path, 'preprocess1', paste0(data, '_', fold, '_pre_valid_', method, '_', di_repair_level, '.csv')))
        dtrain <- read.csv(file.path(res_path, 'preprocess1', paste0(data, '_', fold, '_pre_train_', method, '_', di_repair_level, '.csv')))
      } else {
        dtest  <- read.csv(file.path(res_path, 'preprocess1', paste0(data, '_', fold, '_pre_test_',  method, '.csv')))
        dval   <- read.csv(file.path(res_path, 'preprocess1', paste0(data, '_', fold, '_pre_valid_', method, '.csv')))
        dtrain <- read.csv(file.path(res_path, 'preprocess1', paste0(data, '_', fold, '_pre_train_', method, '.csv')))
      }
      
      # load original test data
      dtest_unscaled <- read.csv(file.path(data_path, 'prepared', paste0(data, '_orig_test.csv')))
      
      # load original train data
      dtrain_unscaled <- read.csv(file.path(data_path, 'prepared', paste0(data, '_orig_', fold, '_train.csv')))
      
      # factor encoding
      dtrain$target         <- as.factor(ifelse(dtrain$target == 2,         'Parole', 'Noparole'))
      dval$target           <- as.factor(ifelse(dval$target   == 2,         'Parole', 'Noparole'))
      dtest$target          <- as.factor(ifelse(dtest$target  == 2,         'Parole', 'Noparole'))
      dtest_unscaled$target <- as.factor(ifelse(dtest_unscaled$target == 2, 'Parole', 'Noparole'))
      dtrain$race           <- as.factor(ifelse(dtrain$race == 1,            'White',  'Racialized'))
      dval$race             <- as.factor(ifelse(dval$race   == 1,            'White',  'Racialized'))
      dtest$race            <- as.factor(ifelse(dtest$race  == 1,            'White',  'Racialized'))
      dtest_unscaled$race   <- as.factor(ifelse(dtest_unscaled$race == 1,    'White',  'Racialized'))
      dtrain$rehab.Not.participated  <- as.factor(ifelse(dtrain$rehab.Not.participated == 1,   'Norehab',  'Rehab'))
      dval$rehab.Not.participated     <- as.factor(ifelse(dval$rehab.Not.participated   == 1,   'Norehab',  'Rehab'))
      dtest$rehab.Not.participated    <- as.factor(ifelse(dtest$rehab.Not.participated  == 1,   'Norehab',  'Rehab'))
      dtest_unscaled$rehab.Not.participated   <- as.factor(ifelse(dtest_unscaled$rehab.Not.participated == 1,    'Norehab',  'Rehab'))
      dtrain$rehab.Participated  <- as.factor(ifelse(dtrain$rehab.Participated == 1,   'Rehab',  'Norehab'))
      dval$rehab.Participated     <- as.factor(ifelse(dval$rehab.Participated   == 1,   'Rehab',  'Norehab'))
      dtest$rehab.Participated    <- as.factor(ifelse(dtest$rehab.Participated  == 1,   'Rehab',  'Norehab'))
      dtest_unscaled$rehab.Participated   <- as.factor(ifelse(dtest_unscaled$rehab.Participated == 1,    'Rehab',  'Norehab'))
      # dtrain <- subset(dtrain, select = -rehab.Participated)
      
      
      #---- TRAINING ----
      
      # grid search params
      source(file.path(func_path, '97_caret_settings.R'))
      source(file.path(func_path, '98_param_grids.R'))
      
      # train models and save result to model.'name'
      for (m in model.names) {
        print(paste0('-- ', m, '...'))
        grid <- get(paste('param.', m, sep = ''))
        args.train <- list(target ~ .,
                           data      = dtrain,
                           method    = m,
                           tuneGrid  = grid,
                           metric    = 'ROC', 
                           trControl = model.control)
        
        args.model <- c(args.train, get(paste('args.', m, sep = '')))
        assign(paste('model.', m, sep = ''), do.call(caret::train, args.model))
        print(paste('-- model', m, 'finished training:', Sys.time(), sep = ' '))
      }
      
      # clean up
      for (m in model.names) {
        rm(list = c(paste0('args.', m), paste0('param.', m)))
      }
      gc()
      rm(args.model, args.train, model.control)
      
      
      #---- THRESHOLDING ----
      
      # Find optimal cutoff based on validation set
      for (m in model.names) {
        pred <- predict(get(paste('model.', m, sep = '')), newdata = dval, type = 'prob')$Parole
        print(paste("Length of validation predictions for model", m, ":", length(pred)))
        print(paste("Length of validation targets for model", m, ":", length(dval$target)))
        EMP  <- empCreditScoring(scores = pred, classes = dval$target)
        assign(paste0('cutoff.', m), EMP$EMPCfrac)
        assign(paste0('EMP.',    m), EMP$EMPC)
      }
      
      
      #---- TESTING ----
      
      # save image
      if (method == 'DI') {
        save.image(file.path(res_path, 'preprocess2', 'intermediate', paste0('IMAGE_PRE_', data, '_', method, '_', fold, '_', di_repair_level, '.Rdata')))
      } else {
        save.image(file.path(res_path, 'preprocess2', 'final', paste0('IMAGE_PRE_', data, '_', method, '_', fold, '.Rdata')))
      }
      
      # reload helper functions
      source(file.path(func_path, '94_evaluate.R'))
      source(file.path(func_path, '95_fairness_metrics.R'))
      source(file.path(func_path, '96_emp_summary.R'))
      source(file.path(func_path, '97_caret_settings.R'))
      source(file.path(func_path, '98_param_grids.R'))
      source(file.path(func_path, '99_compute_profit.R'))
      
      # assess test results
      test_results <- NULL
      for (m in model.names) {
        
        # extract preds and scores
        pred         <- predict(get(paste0('model.', m)), newdata = dtest, type = 'prob')$Parole
        cutoff       <- quantile(pred, get(paste0('cutoff.', m)))
        cutoff_label <- sapply(pred, function(x) ifelse(x > cutoff, 'Parole', 'Noparole'))
        
        # Ensure lengths match
        if (length(cutoff_label) != length(dtest$target)) {
          stop("Length of predicted labels and actual targets do not match.")
        }
        
        # evaluation
        res <- evaluate(class_preds = cutoff_label, 
                        score_preds = pred,
                        targets     = dtest$target,
                        healths     = dtest$sentence,
                        race        = dtest$race,
                        r           = 0.753)
        
        test_results <- cbind(test_results, res)
      }
      
      # save results
      colnames(test_results) <- c(model.names)
      if (method == 'DI') {
        write.csv(test_results, file.path(res_path, 'preprocess2', 'intermediate', paste0(data, '_', fold, '_', method, '_', di_repair_level, '_results.csv')), row.names = TRUE)
      } else {
        write.csv(test_results, file.path(res_path, 'preprocess2', 'final', paste0(data, '_', fold, '_', method, '_results.csv')), row.names = TRUE)
      }
    }
  }
}

# close cluster
stopImplicitCluster()
