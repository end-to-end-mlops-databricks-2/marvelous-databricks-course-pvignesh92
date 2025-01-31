from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.profile("DEFAULT").getOrCreate()
filepath = "/Volumes/mlops_dev/pvignesh/data"

df = spark.read.option("header","true").csv(filepath)
df.display()