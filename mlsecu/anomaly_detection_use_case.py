import pandas as pd
import mlsecu.data_exploration_utils as deu
import mlsecu.data_preparation_utils as dpu
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


def get_list_of_attack_types(dataframe: pd.DataFrame) -> list | None:
    if dataframe is None:
        return None
    df = deu.get_unique_values(dataframe, 'attack_type')
    return df[df != '-']


def get_nb_of_attack_types(dataframe: pd.DataFrame) -> int | None:
    if dataframe is None:
        return None
    return len(get_list_of_attack_types(dataframe))


def get_list_of_if_outliers(dataframe: pd.DataFrame, outlier_fraction: float, random_state=42) -> list | None:
    if dataframe is None:
        return None
    df = dpu.get_one_hot_encoded_dataframe(dataframe)
    df = dpu.remove_nan_through_mean_imputation(df)
    iso_forest = IsolationForest(random_state=random_state, contamination=outlier_fraction)
    y_pred = iso_forest.fit_predict(df)
    return df[y_pred == -1].index.tolist()


def get_list_of_lof_outliers(dataframe: pd.DataFrame, outlier_fraction: float) -> list | None:
    if dataframe is None:
        return None
    df = dpu.get_one_hot_encoded_dataframe(dataframe)
    df = dpu.remove_nan_through_mean_imputation(df)
    lof = LocalOutlierFactor(contamination=outlier_fraction)
    y_pred = lof.fit_predict(df)
    return dataframe[y_pred == -1].index.tolist()


def get_list_of_parameters(dataframe: pd.DataFrame) -> list | None:
    if dataframe is None:
        return None
    return dataframe.columns.tolist()


def get_nb_of_if_outliers(dataframe: pd.DataFrame, outlier_fraction: float, random_state=42) -> int | None:
    if dataframe is None:
        return None
    return len(get_list_of_if_outliers(dataframe, outlier_fraction, random_state))


def get_nb_of_lof_outliers(dataframe: pd.DataFrame, outlier_fraction: float) -> int | None:
    if dataframe is None:
        return None
    return len(get_list_of_lof_outliers(dataframe, outlier_fraction))


def get_nb_of_occurrences(dataframe: pd.DataFrame) -> int | None:
    if dataframe is None:
        return None
    return len(dataframe)


def get_nb_of_parameters(dataframe: pd.DataFrame) -> int | None:
    if dataframe is None:
        return None
    return len(get_list_of_parameters(dataframe))
