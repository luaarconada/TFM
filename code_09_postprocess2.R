###################################
#                                 
#             OVERVIEW            
#                                 
###################################

# This code implements Equalized Odds Processor. The intermediate results are 
# imported from the files saved in `code_08_postprocess1.R`. The code saves final 
# results for equalized odds processor in `results`.



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
data_list <- 'datacorr1'
data_list <- 'datacorr3'
data_out <- 'datacorr1'

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

# Looping over datasets
for (data in data_list) {
  # read data
  dtest_unscaled <- read.csv(file.path(data_path, 'prepared', paste0(data, '_orig_test.csv')))
  dtest_unscaled <- subset(dtest_unscaled, select = c(sentence, race, target))

  # factor encoding
  dtest_unscaled$target <- as.factor(ifelse(dtest_unscaled$target == 1, 'Parole', 'Noparole'))
  dtest_unscaled$race    <- as.factor(ifelse(dtest_unscaled$race == 1,    'White',  'Racialized'))

  # Looping over folds
  #for (fold in seq(0, num_folds - 1)) {
    print('----------------------------------------')
    print(paste0('FOLD: ', fold))
    print('----------------------------------------')
    
    tryCatch({
      # Importing validation data
      dval <- read.csv(file.path(data_path, 'prepared', paste0(data, '_orig_', fold, '_valid.csv')))
      dval$target <- as.factor(ifelse(dval$target == 1, 'Parole', 'Noparole'))
      dval$race <- as.factor(ifelse(dval$race == 1, 'White', 'Racialized'))
      
      # Importing predictions
      val_pred <- read.csv(file.path(res_path, 'in2post1', paste0(data_out, '_', fold, '_AD_POST_training_results_dval.csv')))
      val_pred <- cbind(val_pred, race = dval$race, target = dval$target)
      test_pred <- read.csv(file.path(res_path, 'in2post1', paste0(data_out, '_', fold, '_AD_POST_training_results_dtest.csv')))
      test_pred <- cbind(test_pred, race = dtest_unscaled$race, target = dtest_unscaled$target, sentence = dtest_unscaled$sentence)
      
      # Loading parameter grids
      source(file.path(func_path, '98_param_grids.R'))
      
      # Looping over model names
      for (m in model.names) {
        # Extracting predictions by race
        pred_0 <- val_pred[val_pred$race == 'Racialized', paste0(m, '_scores')]
        pred_1 <- val_pred[val_pred$race == 'White', paste0(m, '_scores')]
        
        # Finding threshold for unprivileged group
        EMP <- empCreditScoring(scores = pred_0, classes = val_pred$target[val_pred$race == 'Racialized'])
        assign(paste0('0_cutoff.', m), EMP$EMPCfrac)
        cutoff_label <- sapply(pred_0, function(x) ifelse(x <= quantile(pred_0, get(paste0('0_cutoff.', m))), 'Noparole', 'Parole'))
        
        # Computing sensitivity
        cm <- confusionMatrix(data = as.factor(cutoff_label), reference = as.factor(val_pred$target[val_pred$race == 'Racialized']))
        sens_0 <- cm$byClass[['Sensitivity']]
        
        # Finding threshold for privileged group
        roc_curve <- roc(val_pred$target[val_pred$race == 'White'], pred_1)
        my.coords <- coords(roc = roc_curve, x = 'all', transpose = F)
        percent <- ecdf(pred_1)
        assign(paste0('1_cutoff.', m), 1 - percent(my.coords[which.min(abs(my.coords$sensitivity - sens_0)), ]$threshold))
        cutoff_label <- sapply(pred_1, function(x) ifelse(x <= quantile(pred_1, get(paste0('1_cutoff.', m))), 'Noparole', 'Parole'))
      }
      
      # Assessing test results
      test_results <- NULL
      for (m in model.names) {
        assign(paste0('0_cutoff.', m), quantile(test_pred[test_pred$race == 'Racialized', paste0(m, '_scores')], get(paste0('0_cutoff.', m))))
        assign(paste0('1_cutoff.', m), quantile(test_pred[test_pred$race == 'White', paste0(m, '_scores')], get(paste0('1_cutoff.', m))))
        cutoff_label_0 <- sapply(test_pred[test_pred$race == 'Racialized', paste0(m, '_scores')], function(x) ifelse(x <= get(paste0('0_cutoff.',m)), 'Noparole', 'Parole'))
        cutoff_label_1 <- sapply(test_pred[test_pred$race == 'White', paste0(m, '_scores')], function(x) ifelse(x <= get(paste0('1_cutoff.',m)), 'Noparole', 'Parole'))
        cutoff_label <- c(cutoff_label_0, cutoff_label_1)
        test_label <- as.factor(c(as.character(test_pred$target[test_pred$race == 'Racialized']), as.character(test_pred$target[test_pred$race == 'White'])))
        sentence <- c(test_pred$sentence[test_pred$race == 'Racialized'], test_pred$sentence[test_pred$race == 'White'])
        race <- c(rep(0, length(cutoff_label_0)), rep(1, length(cutoff_label_1)))
        
        # Evaluation
        res <- evaluate(class_preds = cutoff_label, 
                        score_preds = ifelse(cutoff_label == 'Parole', 1, 0),
                        targets = test_label, 
                        healths = sentence, 
                        race = race,
                        r = 0.753)
        test_results <- cbind(test_results, res)
      }
      
      # Saving test results
      colnames(test_results) <- c(model.names)
      write.csv(test_results, file.path(res_path, 'in2post1post2final', paste0(data_out, '_', fold, '_EOP_AD_results.csv')), row.names = TRUE)
    }, error = function(e) {
      message("Error in processing fold ", fold, " of dataset ", data, ": ", e)
    })
  #}
}
