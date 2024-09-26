import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import joblib
import time
import pandas as pd
import os

class GBMModel():
    """
    Define Class that handles training the h2o GBM, validation and saving the model
    - Train method takes: train, valid, test and predictors
    """
    def __init__(self,  model_name= 'payment_model'):
        self.model_name = model_name
        self.model = None

    def train_model(self, train, valid, test, predictors):
        """Train the model on the provided training data"""
        target = 'outcome'
        self.train = train
        self.test = test
        self.valid = valid
        
        # Define grid of hyperparameters
        gbm_params = {'learn_rate': [j * 0.01 for j in range(1,4)],
                      'max_depth': list(range(3,8)),
                      'sample_rate': [0.8],
                      'col_sample_rate': [0.8,0.9],
                      'ntrees': [100,200],
                      'min_rows': [10,50]}
        
        # Search criteria
        search_criteria = {'strategy': 'RandomDiscrete',
                           'max_models': 10,
                           'seed': 1}

        gbm_params['balance_classes'] = [True] # Oversample the minority class to deal with class imbalance
        search_criteria['stopping_metric'] = 'aucpr' # Area under the precision recall curve is appropriate for imbalanced data

        # Define model
        gbm_grid = H2OGridSearch(model = H2OGradientBoostingEstimator(),
                                 grid_id = 'gbm_grid_1',
                                 hyper_params = gbm_params,
                                 search_criteria = search_criteria)
        
        # Start the timer
        start_time = time.time()

        # Train the model and tune hyperparameters with 3 fold cross-validation and calibrate probabilities
        gbm_grid.train(x = predictors,
                       y = target,
                       training_frame = train,
                       fold_assignment = 'Stratified',
                       nfolds = 3,
                       categorical_encoding = 'enum',
                       stopping_rounds = 5,
                       stopping_tolerance = 1e-4,
                       calibrate_model = True, # apply platt scaling to calibrate predictions
                       calibration_frame = valid,
                       seed = 1)
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"The model training time was {elapsed_time} seconds")

        # Select the best model by choosing the one with the highest AUC PR
        gbm_gridperf = gbm_grid.get_grid(sort_by = 'aucpr', decreasing = True)

        best_gbm = gbm_gridperf.models[0]

        self.model = best_gbm # Save best model as object attribute

    def log_performance(self):
        model = self.model
        train = self.train
        test = self.test

        output_dir = '../Payments/evaluation'
        output_file = os.path.join(output_dir, 'performance.csv')

        train_perf = model.model_performance(train)
        test_perf = model.model_performance(test)

        # Grab the AUC ROC and AUC Precision-Recall for both train and test set
        train_auc_roc = train_perf.auc()
        train_auc_pr = train_perf.aucpr()
        test_auc_roc = test_perf.auc()
        test_auc_pr = test_perf.aucpr()

        # Grab the precision for class 1
        train_precision_class1 = train_perf.precision()[0][1]
        test_precision_class1 = test_perf.precision()[0][1]

        # Grab the recall for class 1
        train_recall_class1 = train_perf.recall()[0][1]
        test_recall_class1 = test_perf.recall()[0][1]
    
        # Grab the precision for class 0
        train_precision_class0 = train_perf.precision()[0][0]
        test_precision_class0 = test_perf.precision()[0][0]

        # Grab the recall for class 0
        train_recall_class0 = train_perf.recall()[0][0]
        test_recall_class0 = test_perf.recall()[0][0]
    
        # Grab the accuracy
        train_accuracy = train_perf.accuracy()[0][1]
        test_accuracy = test_perf.accuracy()[0][1]

        # Grab the f1 score
        train_f1 = train_perf.F1()[0][1]
        test_f1 = test_perf.F1()[0][1]

        # Grab the mean squared error
        train_mse = train_perf.mse()
        test_mse = test_perf.mse()

        # Metrics
        metrics = [
            'train_auc_roc', 'test_auc_roc', 'train_auc_pr', 'test_auc_pr',
            'train_precision_class1', 'train_precision_class0',
            'test_precision_class1', 'test_precision_class0',
            'train_recall_class1', 'train_recall_class0',
            'test_recall_class1', 'test_recall_class0',
            'train_accuracy', 'test_accuracy', 
            'train_f1', 'test_f1',
            'train_mse', 'test_mse'
        ]

        # Metric values
        values = [
            train_auc_roc, test_auc_roc, train_auc_pr, test_auc_pr,
            train_precision_class1, train_precision_class0,
            test_precision_class1, test_precision_class0,
            train_recall_class1, train_recall_class0,
            test_recall_class1, test_recall_class0,
            train_accuracy, test_accuracy,
            train_f1, test_f1,
            train_mse, test_mse
        ]

        # Create the DataFrame using a dictionary
        results = pd.DataFrame({
            'Metric': metrics,
            'Value': values
        })   

        results.to_csv(output_file, index = False) # Save as csv in the performance folder

        return results

    def save_h2o_mojo_model(self):
        model = self.model
        models_dir = '../Payments/models'

        # Save the model as a MOJO
        mojo_path = model.download_mojo(path=models_dir)
        print(f"Model saved as MOJO to: {mojo_path}")

    def save_h2o_model(self):
        model = self.model

        models_dir = '../Payments/models'

        model_path = h2o.save_model(model = model, path = models_dir, force = True)
        print(f"Model saved to: {models_dir}")

    def h2o_shutdown(self):
        h2o.shutdown()