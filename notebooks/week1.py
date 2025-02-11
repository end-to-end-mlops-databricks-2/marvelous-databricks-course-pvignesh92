# Databricks notebook source
# COMMAND ----------
import logging
import sys

import yaml
# from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession
# Add project root to Python path
sys.path.append("./src")
from car_price.config import ProjectConfig
from car_price.data_processor import DataProcessor
from car_price.utils import print_shape, read_csv_pandas

# COMMAND ----------

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

config = ProjectConfig.from_yaml(config_path="./project_config.yml")
logger.info(f"Configuration loaded: {config}")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Reading the datsets
cars_data_df = read_csv_pandas(file_path="./data/used_cars_data.csv")
logger.info(f"Shape of Cars data: {print_shape(cars_data_df)}")

# COMMAND ----------
# Initialize DataProcessor
data_processor = DataProcessor(cars_data_df, config)
# Preprocess the data
data_processor.preprocess()

# COMMAND ----------

# Split the data
X_train, X_test = data_processor.split_data()
logger.info(f"Training set shape: {print_shape(X_train)}")
logger.info(f"Test set shape: {print_shape(X_test)}")

# Cleaning the column names to remove any special characters
X_train_cleaned = data_processor.clean_column_names(X_train)
X_test_cleaned = data_processor.clean_column_names(X_test)

# COMMAND ----------
spark = DatabricksSession.builder.getOrCreate()
data_processor.save_to_catalog(X_train_cleaned, X_test_cleaned, spark, "overwrite")
