from pyspark.sql import SparkSession
import pandas as pd
from typing import Tuple

def read_csv_pandas(file_path:str) -> pd.DataFrame:
    """Read a CSV file into a Pandas DataFrame."""
    return pd.read_csv(file_path)


def read_csv_spark(file_path:str, spark:SparkSession) -> pd.DataFrame:
    """Read a CSV file into a Spark DataFrame."""
    return spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)


def print_shape(df: pd.DataFrame) -> Tuple[int, int]:
    """Return the shape of a DataFrame."""
    print(f"The Dataframe has {df.shape[0]} rows and {df.shape[1]} columns")


def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the correlation matrix of a DataFrame."""
    corr_res = df.corr()
    return corr_res[corr_res >0.2]


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return the missing values in a DataFrame."""
    return df.isnull().sum()