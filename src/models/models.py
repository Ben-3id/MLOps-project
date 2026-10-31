import mlflow
import pandas as pd
from sklearn.linear_model import RANSACRegressor , Ridge , RidgeCV , Lasso , LassoCV , LinearRegression
from sklearn.tree import DecisionTreeRegressor , ExtraTreeRegressor 
from sklearn.ensemble import (GradientBoostingRegressor , RandomForestRegressor ,
                                AdaBoostRegressor , BaggingRegressor ,StackingRegressor )



def get_models(cfg):
    model_dict = {
        "RANSACRegressor" : {"model": RANSACRegressor(estimator=LinearRegression()) , "params": cfg['models']['RANSACRegressor']} , 
        "Ridge" : {"model": Ridge() , "params": cfg['models']['Ridge']},
        "RidgeCV" : {"model": RidgeCV() , "params": cfg['models']['RidgeCV'] } ,
        "Lasso" : {"model": Lasso() , "params": cfg['models']['Lasso']},
        "LassoCV": {"model": LassoCV() , "params": cfg['models']['LassoCV']},

        "DecisionTreeRegressor" : {"model": DecisionTreeRegressor() , "params": cfg['models']['DecisionTreeRegressor']},
        "ExtraTreeRegressor" : {"model": ExtraTreeRegressor() , "params": cfg['models']['ExtraTreeRegressor'] },

        "GradientBoostingRegressor" : {"model": GradientBoostingRegressor() , "params": cfg['models']['GradientBoostingRegressor']},
        "RandomForestRegressor" : {"model": RandomForestRegressor() , "params": cfg['models']['RandomForestRegressor']} ,

        "AdaBoostRegressor" : {"model": AdaBoostRegressor() , "params": cfg['models']['AdaBoostRegressor']},
        "BaggingRegressor" : {"model": BaggingRegressor() , "params": cfg['models']['BaggingRegressor']},
    }
    return model_dict