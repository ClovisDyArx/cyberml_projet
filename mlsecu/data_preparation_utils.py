import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def get_one_hot_encoded_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame | None:
    if dataframe is None:
        return None
    return pd.get_dummies(dataframe)


def remove_nan_through_mean_imputation(dataframe: pd.DataFrame) -> pd.DataFrame | None:
    if dataframe is None:
        return None
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    return pd.DataFrame(imp_mean.fit_transform(dataframe), columns=dataframe.columns)
