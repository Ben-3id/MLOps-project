from sklearn.model_selection import train_test_split
import pandas as pd
from ..settings import load_config
from ..utilities import Preprocess
from pathlib import Path
import os

def main():
    root_path = Path(__file__).parent.parent

    config = load_config()

    dist_path = os.path.join(
        root_path, 
        config['data']['processed_data_path']
        )

    X = pd.read_csv(
        os.path.join(
            os.path.join(
                root_path , config['data']['selected_data_path']
            ) , "Selected_X.csv"
        )
    )

    y = pd.read_csv(
        os.path.join(
            os.path.join(
                root_path , config['data']['selected_data_path']
            ) , "Selected_y.csv"
        )
    )

    X_train , X_test , y_train , y_test = train_test_split(
                X , y ,
                test_size = config['data']['test_size'] ,
                shuffle= config['data']['shuffle'] ,
                random_state= config['data']['random_state']
        )

    if not os.path.exists(dist_path):
        os.makedirs(dist_path)

    processor = Preprocess(config , X_train.sample(5))

    processor.fit(X_train , y_train)

    X_train_proced , y_train_proced  = processor.transform(X_train ,y_train)

    X_train_proced.to_csv(os.path.join(dist_path , "X_train_proced.csv"),index= False)
    
    y_train_proced.to_csv(os.path.join(dist_path , "y_train_proced.csv"),index= False)

    X_test_proced  , y_test_proced  = processor.transform( X_test , y_test )

    X_test_proced.to_csv(os.path.join(dist_path , "X_test_proced.csv") ,index= False)
    
    y_test_proced.to_csv(os.path.join(dist_path , "y_test_proced.csv") ,index= False)




if __name__ == "__main__":
    main()