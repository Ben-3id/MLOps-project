import mlflow
import pandas as pd
from sklearn.linear_model import RANSACRegressor , Ridge , RidgeCV , Lasso , LassoCV
from sklearn.tree import DecisionTreeRegressor , ExtraTreeRegressor 
from sklearn.ensemble import (GradientBoostingRegressor , RandomForestRegressor ,
                                AdaBoostRegressor , BaggingRegressor ,StackingRegressor )



def get_models():
    model_dict = {
        "RANSACRegressor" : RANSACRegressor() , 
        "Ridge" : Ridge() ,
        "RidgeCV" : RidgeCV() ,
        "Lasso" : Lasso(),
        "LassoCV": LassoCV(),

        "DecisionTreeRegressor" : DecisionTreeRegressor(),
        "ExtraTreeRegressor" : ExtraTreeRegressor(),

        "GradientBoostingRegressor" : GradientBoostingRegressor(),
        "RandomForestRegressor" : RandomForestRegressor() ,

        "AdaBoostRegressor" : AdaBoostRegressor(),
        "BaggingRegressor" : BaggingRegressor(),
        "StackingRegressor" :StackingRegressor(estimators=[ ("Lasso", Lasso()),
                                                        ("Ridge" , Ridge()) ] )
    }
    return model_dict