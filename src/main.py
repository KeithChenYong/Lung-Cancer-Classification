import sys
import os
import configparser
from data_ingestion import load_df
from data_processing import original_df
from model import train_model
from evaluate import evaluate_model

def load_config(config_path):
    """Load configurations from the specified config file."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def main():
    ''' Main python script to execute loading of data, propressing, model, cross validation, evaluating'''
    # Load data
    current_dir = os.getcwd()
    
    # Define the relative path to cancer_patient_data_sets.db based on the current directory
    if "src" in current_dir:
        db_path = "../data/cancer_patient_data_sets.db"
        config_path = "../config.ini"
    else:
        db_path = "data/cancer_patient_data_sets.db"
        config_path = "config.ini"       

    # Updated version on 04Apr24
    if os.path.exists(db_path):
        df = load_df(db_path, "SELECT * FROM cancer_data")
    else:
        print(f"Error: Database file '{db_path}' not found. Please check your data file path.")
        return  # Exit the function or handle the error as needed

    # Explore, clean & preprocess + Dimension reduction & feature engineering
    X, y = original_df(df)

    # Load configurations from the config file
    config = load_config(config_path)
    
    # Train and evaluate models
    model, X_test, y_test = train_model(X, y, config)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()