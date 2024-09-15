###################################
#                                 
#             OVERVIEW            
#                                 
###################################

# This code implements a set of base classifiers using original data sets. Processed
# and partitioned data are imported from the files saved in `code_00_partitioning.ipynb`. 
# The code saves intermediate results in `results`.


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
data_list <- paste0('data', 1)

# partitioning
num_folds <- 10
seed      <- 1

# cores
cores <- 8

# options
set.seed(seed)
options(scipen = 10)

# parallel computing
nrOfCores <- cores
registerDoParallel(cores = nrOfCores)
message(paste('\n Registered number of cores:\n',nrOfCores,'\n'))



##################################
#                                
#          MODELING      
#                                
##################################

# helper functions
source(file.path(func_path, '95_fairness_metrics.R'))
source(file.path(func_path, '96_emp_summary.R'))

all_lambda = c('0.5','0.6','0.7','0.8','0.9','1.0')

# data loop
for (data in data_list) {
  #for (lambda in all_lambda){
  # modeling
    for (fold in seq(0, num_folds - 1)) {
      
      #---- PREPARATIONS ----
      
      # feedback
      print('----------------------------------------')
      print(paste0('FOLD: ', fold))
      print('----------------------------------------')
      
      disco_path <- "E:/resultados/results/"
      
      # read data
      dtest  <- read.csv(file.path(disco_path, 'preprocess1', paste0(data, '_', fold, '_pre_test_RW.csv')))
      dval   <- read.csv(file.path(disco_path, 'preprocess1', paste0(data, '_', fold, '_pre_valid_RW.csv')))
      dtrain <- read.csv(file.path(disco_path, 'preprocess1', paste0(data, '_', fold, '_pre_train_RW.csv')))
      
      # factor encoding
      dtrain$target         <- as.factor(ifelse(dtrain$target == 2,         'Parole', 'Noparole'))
      dval$target           <- as.factor(ifelse(dval$target   == 2,         'Parole', 'Noparole'))
      dtest$target          <- as.factor(ifelse(dtest$target  == 2,         'Parole', 'Noparole'))
      dtrain$race           <- as.factor(ifelse(dtrain$race == 1,            'White',  'Racialized'))
      dval$race             <- as.factor(ifelse(dval$race   == 1,            'White',  'Racialized'))
      dtest$race            <- as.factor(ifelse(dtest$race  == 1,            'White',  'Racialized'))
      dtrain$rehab.Not.participated  <- as.factor(ifelse(dtrain$rehab.Not.participated == 1,   'Norehab',  'Rehab'))
      dval$rehab.Not.participated     <- as.factor(ifelse(dval$rehab.Not.participated   == 1,   'Norehab',  'Rehab'))
      dtest$rehab.Not.participated    <- as.factor(ifelse(dtest$rehab.Not.participated  == 1,   'Norehab',  'Rehab'))
      dtrain$rehab.Participated  <- as.factor(ifelse(dtrain$rehab.Participated == 1,   'Rehab',  'Norehab'))
      dval$rehab.Participated     <- as.factor(ifelse(dval$rehab.Participated   == 1,   'Rehab',  'Norehab'))
      dtest$rehab.Participated    <- as.factor(ifelse(dtest$rehab.Participated  == 1,   'Rehab',  'Norehab'))
      
      
      #---- TRAINING ----
      
      # grid search params
      source(file.path(func_path, '97_caret_settings.R'))
      source(file.path(func_path, '98_param_grids.R'))
      
      # train models and save result to model.'name'
      for (m in model.names) {
        print(paste0('-- ', m, '...'))
        grid <- get(paste('param.', m, sep = ''))
        args.train <- list(target~., 
                           data      = dtrain,  
                           method    = m, 
                           tuneGrid  = grid,
                           metric    = 'EMP',
                           trControl = model.control)
        
        args.model <- c(args.train, get(paste('args.', m, sep = '')))
        assign(paste('model.', m, sep = ''), do.call(train, args.model))
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
        EMP  <- empCreditScoring(scores = pred, classes = dval$target)
        assign(paste0('cutoff.', m), EMP$EMPCfrac)
      }
      
      
      #---- TRAINING RESULTS ----
      
      # save image
      save.image(file.path(res_path, 'pre1post1', paste0('IMAGE_POST_', data, '_', fold, '_RW_', '.Rdata')))
      
      # loop through data subsets
      data.names <- c('dval', 'dtest')
      for (data.set in data.names) {
        
        # placeholders
        model_prediction <- NULL
        cnames           <- NULL
        
        # loop through models
        for (m in model.names) {
          
          # produce predictions
          pred         <- predict(get(paste0('model.', m)), newdata = get(data.set), type = 'prob')$Parole
          cutoff       <- quantile(pred, get(paste0('cutoff.', m)))
          cutoff_label <- sapply(pred, function(x) ifelse(x > cutoff, 'Parole', 'Noparole'))
          
          # save predictions      
          model_prediction <- cbind(model_prediction, pred, cutoff_label)
          cnames <- c(cnames, c(paste0(m, '_scores'), paste0(m, '_class')))
        }
        
        # export results
        colnames(model_prediction) <- cnames
        write.csv(model_prediction, file.path(res_path, 'pre1post1', paste0(data, '_', fold, '_RW_', '_POST_training_results_', data.set, '.csv')), row.names = F)
      }
    }
  #}
}

# close cluster
stopImplicitCluster()
