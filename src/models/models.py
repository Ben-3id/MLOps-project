
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    Perceptron
)

from sklearn.tree import (
    DecisionTreeClassifier,
    ExtraTreeClassifier
)

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


def get_models(cfg):

    models = {
        "LogisticRegression": LogisticRegression(),
        "RidgeClassifier": RidgeClassifier(),
        "SGDClassifier": SGDClassifier(),
        "Perceptron": Perceptron(),

        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "ExtraTreeClassifier": ExtraTreeClassifier(),

        "GradientBoostingClassifier": GradientBoostingClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
        "ExtraTreesClassifier": ExtraTreesClassifier(),

        "AdaBoostClassifier": AdaBoostClassifier(),
        "BaggingClassifier": BaggingClassifier(),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(),

        "SVC": SVC(),
        "LinearSVC": LinearSVC(),

        "KNeighborsClassifier": KNeighborsClassifier(),

        "GaussianNB": GaussianNB(),
        "BernoulliNB": BernoulliNB(),
    }

    model_dict = {}

    for name, model in models.items():

        params = cfg["models"].get(name, {})

        # Model Only
        model_dict[name] = {
            "model": model,
            "params": params
        }

        # Model + SMOTE
        model_dict[f"{name}_SMOTE"] = {
            "model": Pipeline([
                ("smote", SMOTE(random_state=42)),
                ("model", model.__class__())
            ]),
            "params": {
                f"model__{key}": value
                for key, value in params.items()
            }
        }

    return model_dict
