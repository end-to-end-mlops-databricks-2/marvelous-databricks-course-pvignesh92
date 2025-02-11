# Databricks notebook source
# Add project root to Python path
import sys
from loguru import logger
import mlflow
# from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession

# Add project root to Python path
sys.path.append("./src")
from car_price.config import ProjectConfig, Tags
from car_price.models.basic_lr_model import LinearRegressionModel

# COMMAND ----------
# Default profile:
logger.info("Starting the mlflow tracking process")
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
logger.info("Ending the mlflow tracking process")


# Profile called "course"
# mlflow.set_tracking_uri("databricks://course")
# mlflow.set_registry_uri("databricks-uc://course")
logger.info("Loading the configuration")
config = ProjectConfig.from_yaml(config_path="./project_config.yml")
logger.info("Creating the SparkSession")
# spark = SparkSession.builder.getOrCreate()
spark = DatabricksSession.builder.getOrCreate()

tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

# COMMAND ----------
# Initialize model with the config path
logger.info("Initializing model with the config path")
basic_model = LinearRegressionModel(config=config, tags=tags, spark=spark)

# COMMAND ----------
logger.info("Loading the data to the model")
basic_model.load_data()
logger.info("Preparing the features to fit the model")
basic_model.prepare_features()

# COMMAND ----------
# Train + log the model (runs everything including MLflow logging)
logger.info("Training the model")
basic_model.train()
logger.info("Logging the model")
basic_model.log_model()

# COMMAND ----------
logger.info("Getting the run_id")
run_id = mlflow.search_runs(
    experiment_names=["/Shared/car-price-basic"], filter_string="tags.branch='week2'"
).run_id[0]
logger.info(f"Run ID: {run_id}")
model = mlflow.sklearn.load_model(f"runs:/{run_id}/lr-pipeline-model")

# COMMAND ----------
# Retrieve dataset for the current run
logger.info(f"Run ID: {run_id}")
basic_model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
basic_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
basic_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = basic_model.load_latest_model_and_predict(X_test)
# COMMAND ----------
