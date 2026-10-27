from sklearn.model_selection import train_test_split
import pandas as pd
from ..settings import load_config
from ..utilities import Fillter
from pathlib import Path
import os


def main():
    root_path = Path(__file__).parent.parent

    config = load_config()
    
    threshold = config['data']['threshold']

    dist_path = os.path.join(
        root_path, 
        config['data']['selected_data_path']
        )

    if not os.path.exists(dist_path):
        os.makedirs(dist_path)

    X = pd.read_csv(
        os.path.join(
            os.path.join(
                root_path , config['data']['splitted_data_path']
            ) , "X_df.csv"
        )
    )

    y = pd.read_csv(
        os.path.join(
            os.path.join(
                root_path , config['data']['splitted_data_path']
            ) , "y_df.csv"
        )
    )

    NewX = Fillter(X , y , threshold= threshold)

    NewX.to_csv(
        os.path.join(
            dist_path , "Selected_X.csv"
        )
    ,index= False )

    y.to_csv(
        os.path.join(
            dist_path , "Selected_y.csv"
        )
    ,index= False )


if __name__ == "__main__":
    main()