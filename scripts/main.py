from src.data_handler.data_loader import DataLoader
from src.modelling.model import GBMModel
import logging
import argparse

### Main script to be run in the terminal from the 'Payments directory'
def main():
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        filename = 'modelling.log',
        filemode = 'a')

    # Parse arguments from the terminal
    parser = argparse.ArgumentParser(description = "Run the data processing and modelling pipeline and fit h2o gbm model")

    # Define the argument for predictors
    parser.add_argument('--predictors', nargs='+', 
                        help='List of predictor variables (space-separated)',
                        required=False,
                        default = ['currency_group', 'api_version', 'vertical_group', 'connectivity_type', 'country_group', 'amount_in_currency', 'duration', 'created_day_of_week', 'createdat_ts'])

    args = parser.parse_args()

    # Grab predicts argument defined in the terminal or default predictors if blank
    predictors = args.predictors

    # Run data processing and modelling pipeline
    try:
        logging.info("Loading data...")
        # Load and preprocess the data
        data_loader = DataLoader('truelayer_data_sc_test_data_set.csv', predictors)
        data_loader.load_data()

        logging.info("/n Performing data processing and feature engineering...")
        data_loader.preprocess_data()

        # Split into train, validation and test sets
        data_loader.generate_h2o_train_test()

        # Grab train, valid and test attributes from data_loader object ready for modelling
        train = data_loader.train
        valid = data_loader.valid
        test = data_loader.test

        logging.info("/nFitting GBM...")
        # Train the model
        model = GBMModel()
        model.train_model(train, valid, test, predictors)
        logging.info("/n Saving h2o model in models directory...")
        model.save_h2o_model()
        logging.info("/n Performing validation...")
        model.log_performance()
        model.h2o_shutdown()
        logging.info("/n Pipeline has finished shutting down resources...")
        print("End of process...")

    except Exception as e:
        logging.exception(f"An error occurred: {e}")

if __name__ == '__main__':
    main()