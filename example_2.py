
import sklearn.datasets
from sklearn.model_selection import KFold

import optuna.integration.lightgbm as lgb


if __name__ == "__main__":
    data, target = sklearn.datasets.fetch_kddcup99(return_X_y=True)
    dtrain = lgb.Dataset(data, label=target)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1
    }

    tuner = lgb.LightGBMTunerCV(
        params,
        dtrain,
        time_budget=60,
        verbose_eval=0,
        early_stopping_rounds=50,
        folds=KFold(n_splits=3)
    )

    tuner.run()

    print("Best score:", tuner.best_score)
    best_params = tuner.best_params
    print("Best params:", best_params)
    print("  Params: ")
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))
