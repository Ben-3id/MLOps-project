from sklearn.model_selection import train_test_split
import pandas as pd
from ..settings import load_config
from ..utilities import get_datasets
from pathlib import Path
import os

def main():
    root_path = Path(__file__).parent.parent

    config = load_config()
    paths = get_datasets(config)

    dist_path = os.path.join(
        root_path, 
        config['data']['splitted_data_path']
        )

    if not os.path.exists(dist_path):
        os.makedirs(dist_path)

    df_list = [pd.read_csv(path , sep=";") for path in paths]

    if not df_list:
        raise ValueError("No CSV files found.")


    if len(df_list) == 1:
        data = df_list[0]   
    else: data = pd.concat(
            df_list , ignore_index=True
        )

    if config['data']['ignored_col'] != None:
        data.drop(columns=[config['data']['ignored_col']] , inplace=True)

    X = data.drop(columns=[config['data']['target']])
    
    y = data[config['data']['target']]

    X.to_csv(
        os.path.join(
            dist_path, 
            "X_df.csv") ,index= False
            )

    y.to_csv(
        os.path.join(
            dist_path, 
            "y_df.csv") ,index= False
            )

if __name__ == "__main__":
    main()