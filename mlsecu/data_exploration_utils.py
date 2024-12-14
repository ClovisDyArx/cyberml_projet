import pandas as pd


def get_column_names(dataframe: pd.DataFrame) -> list | None:
    if dataframe is None:
        return None
    return dataframe.columns.tolist()


def get_nb_of_dimensions(dataframe: pd.DataFrame) -> int | None:
    if dataframe is None:
        return None
    return len(dataframe.columns)


def get_nb_of_rows(dataframe: pd.DataFrame) -> int | None:
    if dataframe is None:
        return None
    return dataframe.shape[0]


def get_number_column_names(dataframe: pd.DataFrame) -> list | None:
    if dataframe is None:
        return None
    return dataframe.select_dtypes(include='number').columns.tolist()


def get_object_column_names(dataframe: pd.DataFrame) -> list | None:
    if dataframe is None:
        return None
    return dataframe.select_dtypes(include='object').columns.tolist()


def get_unique_values(dataframe: pd.DataFrame, column_name: str) -> list | None:
    if dataframe is None or column_name is None:
        return None
    return dataframe[column_name].unique()
