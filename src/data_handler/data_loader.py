import pandas as pd
import numpy as np
import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.frame import H2OFrame
import os

class DataLoader():
    def __init__(self, file_name, predictors):
         # Construct the path relative to the current working directory
        self.file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data', file_name)

        self.data = None
        self.predictors = predictors
        h2o.init(nthreads=-1)

    def load_data(self):
        """Load data from local csv file"""
        print(f"Looking for file at: {self.file_path}")  
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file was not found: {self.file_path}")
        
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def preprocess_data(self):
        """Perform data preprocessing
        - Load the data
        - Combine the target into executed vs other, excluding pending payments
        - Perform feature engineering - combining sparse categories, create new column duration (different between lastupdate and createdat)
        - Overwrite data attribute with processed data and also return
        """
        data = self.data

        # Set outcome to be restricted to completed payments to simplify problem
        data = data.loc[data["status"].isin(['Executed', 'failed', 'Failed', 'Cancelled', 'Rejected']), :].copy()
        data['outcome'] = np.where(data['status'] == 'Executed', 'Executed', 'Other') # Dichotomise target
        
        datetime_columns = ['createdat_ts', 'initiated_at', 'authorizing_at', 'authorized_at', 'settled_at', 'lastupdatedat_ts', 'executed_at', 'failed_at']
        for col in datetime_columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')  # Coerce errors to NaT

        object_cols = data.select_dtypes(include='object').columns # Selet object columns

        # object cols is a list of categorical columns - fill with 'Missing'
        data[object_cols] = data[object_cols].fillna('Missing')

        data['duration'] = (data['lastupdatedat_ts'] - data['createdat_ts']).dt.total_seconds() # Duration in seconds

        data = data.query("duration >= 0").copy() # Drop the few rows where duration is negative

        # Combine GBP and EUR into a single category 'EUR_GBP'
        data['currency_group'] = data['currency'].replace({'GBP': 'EUR_GBP', 'EUR': 'EUR_GBP'})

        # Group smaller categories into 'other_vertical'
        data['vertical_group'] = data['vertical'].replace({
            'vertical 1': 'other_vertical',
            'vertical 3': 'other_vertical',
            'Missing': 'other_vertical'
        })

        # Get the frequency of each country_id for combining
        country_counts = data['country_id'].value_counts()

        # Define a threshold and mark less frequent categories as 'other_country'
        common_countries = country_counts[country_counts > 3500].index
        data['country_group'] = data['country_id'].where(data['country_id'].isin(common_countries), 'other_country')

        data['created_day_of_week'] = data['createdat_ts'].dt.dayofweek
    
        self.data = data
        return data
    
    def generate_h2o_train_test(self):
        """Convert data to h2o dataframe ready for modelling, subsetting to predictor and target columns"""
        data = self.data
        predictors = self.predictors
        target = 'outcome'
        cols = predictors + [target]
        h2o_data = H2OFrame(data[cols])

        seed = 45 # set seed for reproducibility

        # Split into train, valid and test set (valid is used to calibrate probabilities) - h2o performs stratified sampling under the hood
        train, valid, test = h2o_data.split_frame(ratios=[0.7, 0.15], seed = seed)

        self.train = train
        self.valid = valid
        self.test = test

        return train, valid, test

    def get_preprocessed_data(self):
        return self.data
    
    