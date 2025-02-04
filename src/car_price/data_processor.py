import logging
import pandas as pd
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split
from car_price.config import ProjectConfig

# Add project root to Python path
sys.path.append("src")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config:ProjectConfig):
        self.df = pandas_df
        self.config = config


    def preprocess(self):
        """Preprocess the DataFrame stored in self.df"""
        # Checking for duplicate values and handling it
        logger.info(f"Shape of data before removing duplicates: {self.df.shape}")
        
        if self.df.duplicated().sum() > 0:
            self.df = self.df.drop_duplicates()

        # Perform imputation of missing values
        cols_list = self.config.features_to_impute
        print(cols_list)
        for col in cols_list:
            self.df[col] = self.df.groupby(["Brand", "Model"])[col].transform(
                lambda x: x.fillna(x.median())
            )
        self.df = self.df[self.df["Price"].notna()].copy()
        logger.info(f"Shape of data after removing duplicates: {self.df.shape}")

        self.df = pd.get_dummies(
                self.df,
                columns=self.df[self.config.cat_features].columns.tolist(),
                drop_first=True,
            )

        logger.info(f"Shape of data after encoding: {self.df.shape}")
        return self.df


    def split_data(self, test_size=0.2, random_state=42):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set


    def clean_column_names(self, df: pd.DataFrame ):
        df.columns = [col.replace(' ', '_').replace(';', '_').replace('{', '_').replace('}', '_')
                    .replace('(', '_').replace(')', '_').replace('\n', '_').replace('\t', '_')
                    .replace('=', '_') for col in df.columns]
        return df
    

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession, write_mode="append"):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))   
        
        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))

        train_set_with_timestamp.write.mode(write_mode).saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set")
        
        test_set_with_timestamp.write.mode(write_mode).saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set")

        spark.sql(f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
        
        spark.sql(f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

       