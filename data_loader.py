import pandas as pd


def load_datasets(
    train_path="training_meteo_data_cleaned.csv",
    val_path="validation_meteo_data_cleaned.csv",
    test_path="testing_meteo_data_cleaned.csv",
):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df
