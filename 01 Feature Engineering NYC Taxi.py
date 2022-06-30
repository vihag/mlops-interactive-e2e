# Databricks notebook source
# MAGIC %md
# MAGIC ## NYC Taxi Fare Prediction: Feature Engineering
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step1.png?raw=true">

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Store taxi example notebook
# MAGIC 
# MAGIC This notebook illustrates the use of Feature Store to create a model that predicts NYC Yellow Taxi fares. It includes these steps:
# MAGIC 
# MAGIC - Compute and write features.
# MAGIC - Train a model using these features to predict fares.
# MAGIC - Evaluate that model on a new batch of data using existing features, saved to Feature Store.
# MAGIC 
# MAGIC ## Requirements
# MAGIC - Databricks Runtime for Machine Learning 8.3 or above.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi_example_flow_v3.png"/>

# COMMAND ----------

# MAGIC %run ./helpers/set_experiment

# COMMAND ----------

# MAGIC %md ## Compute features

# COMMAND ----------

# MAGIC %md #### Load the raw data used to compute features
# MAGIC 
# MAGIC Load the `nyc-taxi-tiny` dataset.  This was generated from the full [NYC Taxi Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) which can be found at `dbfs:/databricks-datasets/nyctaxi` by applying the following transformations:
# MAGIC 
# MAGIC 1. Apply a UDF to convert latitude and longitude coordinates into ZIP codes, and add a ZIP code column to the DataFrame.
# MAGIC 1. Subsample the dataset into a smaller dataset based on a date range query using the `.sample()` method of the Spark `DataFrame` API.
# MAGIC 1. Rename certain columns and drop unnecessary columns.
# MAGIC 
# MAGIC If you want to create this dataset from the raw data yourself, follow these steps:
# MAGIC 1. Run the Feature Store taxi example dataset notebook ([AWS](https://docs.databricks.com/_static/notebooks/machine-learning/feature-store-taxi-example-dataset.html)|[Azure](https://docs.microsoft.com/azure/databricks/_static/notebooks/machine-learning/feature-store-taxi-example-dataset.html)|[GCP](https://docs.gcp.databricks.com/_static/notebooks/machine-learning/feature-store-taxi-example-dataset.html)) to generate the Delta table.
# MAGIC 1. In this notebook, replace the following `spark.read.format("delta").load("/databricks-datasets/nyctaxi-with-zipcodes/subsampled")` with: `spark.read.table("feature_store_taxi_example.nyc_yellow_taxi_with_zips")`

# COMMAND ----------

raw_data = spark.read.format("delta").load("/databricks-datasets/nyctaxi-with-zipcodes/subsampled")
display(raw_data)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC From the taxi fares transactional data, we will compute two groups of features based on trip pickup and drop off zip codes.
# MAGIC 
# MAGIC #### Pickup features
# MAGIC 1. Count of trips (time window = 1 hour, sliding window = 15 minutes)
# MAGIC 1. Mean fare amount (time window = 1 hour, sliding window = 15 minutes)
# MAGIC 
# MAGIC #### Drop off features
# MAGIC 1. Count of trips (time window = 30 minutes)
# MAGIC 1. Does trip end on the weekend (custom feature using python code)
# MAGIC 
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi_example_computation_v5.png"/>

# COMMAND ----------

# MAGIC %md ### Helper functions

# COMMAND ----------

from databricks import feature_store
from pyspark.sql.functions import *
from pyspark.sql.types import FloatType, IntegerType, StringType
from pytz import timezone


@udf(returnType=IntegerType())
def is_weekend(dt):
    tz = "America/New_York"
    return int(dt.astimezone(timezone(tz)).weekday() >= 5)  # 5 = Saturday, 6 = Sunday
  
@udf(returnType=StringType())  
def partition_id(dt):
    # datetime -> "YYYY-MM"
    return f"{dt.year:04d}-{dt.month:02d}"


def filter_df_by_ts(df, ts_column, start_date, end_date):
    if ts_column and start_date:
        df = df.filter(col(ts_column) >= start_date)
    if ts_column and end_date:
        df = df.filter(col(ts_column) < end_date)
    return df


# COMMAND ----------

# MAGIC %md ### Data scientist's custom code to compute features

# COMMAND ----------

def pickup_features_fn(df, ts_column, start_date, end_date):
    """
    Computes the pickup_features feature group.
    To restrict features to a time range, pass in ts_column, start_date, and/or end_date as kwargs.
    """
    df = filter_df_by_ts(
        df, ts_column, start_date, end_date
    )
    pickupzip_features = (
        df.groupBy(
            "pickup_zip", window("tpep_pickup_datetime", "1 hour", "15 minutes")
        )  # 1 hour window, sliding every 15 minutes
        .agg(
            mean("fare_amount").alias("mean_fare_window_1h_pickup_zip"),
            count("*").alias("count_trips_window_1h_pickup_zip"),
        )
        .select(
            col("pickup_zip").alias("zip"),
            unix_timestamp(col("window.end")).alias("ts").cast(IntegerType()),
            partition_id(to_timestamp(col("window.end"))).alias("yyyy_mm"),
            col("mean_fare_window_1h_pickup_zip").cast(FloatType()),
            col("count_trips_window_1h_pickup_zip").cast(IntegerType()),
        )
    )
    return pickupzip_features
  
def dropoff_features_fn(df, ts_column, start_date, end_date):
    """
    Computes the dropoff_features feature group.
    To restrict features to a time range, pass in ts_column, start_date, and/or end_date as kwargs.
    """
    df = filter_df_by_ts(
        df,  ts_column, start_date, end_date
    )
    dropoffzip_features = (
        df.groupBy("dropoff_zip", window("tpep_dropoff_datetime", "30 minute"))
        .agg(count("*").alias("count_trips_window_30m_dropoff_zip"))
        .select(
            col("dropoff_zip").alias("zip"),
            unix_timestamp(col("window.end")).alias("ts").cast(IntegerType()),
            partition_id(to_timestamp(col("window.end"))).alias("yyyy_mm"),
            col("count_trips_window_30m_dropoff_zip").cast(IntegerType()),
            is_weekend(col("window.end")).alias("dropoff_is_weekend"),
        )
    )
    return dropoffzip_features  

# COMMAND ----------

from datetime import datetime

pickup_features = pickup_features_fn(
    raw_data, ts_column="tpep_pickup_datetime", start_date=datetime(2016, 1, 1), end_date=datetime(2016, 1, 31)
)
dropoff_features = dropoff_features_fn(
    raw_data, ts_column="tpep_dropoff_datetime", start_date=datetime(2016, 1, 1), end_date=datetime(2016, 1, 31)
)

# COMMAND ----------

display(pickup_features)

# COMMAND ----------

# MAGIC %md ### Use Feature Store library to create new feature tables 

# COMMAND ----------

# MAGIC %md First, create the database where the feature tables will be stored.

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE DATABASE IF NOT EXISTS vg_mlops_cicd_db;

# COMMAND ----------

# MAGIC %md Next, create an instance of the Feature Store client.

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md
# MAGIC Use the `create_feature_table` API to define schema and unique ID keys. If the optional `features_df` argument is passed, the API also writes the data to Feature Store.

# COMMAND ----------

sqlContext.setConf("spark.sql.shuffle.partitions", "5")

fs.create_feature_table(
    name="vg_mlops_cicd_db.trip_pickup_features",
    keys=["zip", "ts"],
    schema=pickup_features.schema,
    partition_columns="yyyy_mm",
    description="Taxi Fares. Pickup Features",
)
fs.create_feature_table(
    name="vg_mlops_cicd_db.trip_dropoff_features",
    keys=["zip", "ts"],
    schema=dropoff_features.schema,
    partition_columns="yyyy_mm",
    description="Taxi Fares. Dropoff Features",
)

fs.write_table(
  name="vg_mlops_cicd_db.trip_pickup_features",
  df=pickup_features,
  mode="overwrite"
)

fs.write_table(
  name="vg_mlops_cicd_db.trip_dropoff_features",
  df=dropoff_features,
  mode="merge"
)

# COMMAND ----------

# MAGIC %md ## Update features
# MAGIC 
# MAGIC Use the `compute_and_write` function to update the feature table values. This Feature Store function is an attribute of user-specified functions that are decorated with `@feature_store.feature_table`.
# MAGIC 
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi_example_compute_and_write.png"/>

# COMMAND ----------

display(raw_data)

# COMMAND ----------

# Compute the pickup_features feature group.
pickup_features_df = pickup_features_fn(
  df=raw_data,
  ts_column="tpep_pickup_datetime",
  start_date=datetime(2016, 2, 1),
  end_date=datetime(2016, 2, 29),
)

# Write the pickup features DataFrame to the feature store table
fs.write_table(
  name="vg_mlops_cicd_db.trip_pickup_features",
  df=pickup_features_df,
  mode="merge",
)

# Compute the dropoff_features feature group.
dropoff_features_df = dropoff_features_fn(
  df=raw_data,
  ts_column="tpep_dropoff_datetime",
  start_date=datetime(2016, 2, 1),
  end_date=datetime(2016, 2, 29),
)

# Write the dropoff features DataFrame to the feature store table
fs.write_table(
  name="vg_mlops_cicd_db.trip_dropoff_features",
  df=dropoff_features_df,
  mode="merge",
)

# COMMAND ----------

# MAGIC %md When writing, both `merge` and `overwrite` modes are supported.
# MAGIC 
# MAGIC     dropoff_features_fn.compute_and_write(
# MAGIC         input={
# MAGIC           'df': raw_data, 
# MAGIC           'ts_column': "tpep_dropoff_datetime", 
# MAGIC           'start_date': datetime(2016, 2, 1),
# MAGIC           'end_date': datetime(2016, 2, 29),
# MAGIC         },
# MAGIC         feature_table_name="feature_store_taxi_example.trip_dropoff_features",
# MAGIC         mode="overwrite"
# MAGIC     )
# MAGIC 
# MAGIC 
# MAGIC Data can also be streamed into Feature Store with `compute_and_write_streaming`, for example:
# MAGIC 
# MAGIC     dropoff_features_fn.compute_and_write_streaming(
# MAGIC         input={
# MAGIC           'df': streaming_input, 
# MAGIC           'ts_column': "tpep_dropoff_datetime", 
# MAGIC           'start_date': datetime(2016, 2, 1),
# MAGIC           'end_date': datetime(2016, 2, 29),
# MAGIC         },
# MAGIC         feature_table_name="feature_store_taxi_example.trip_dropoff_features",
# MAGIC     )
# MAGIC     
# MAGIC You can schedule a notebook to periodically update features using Databricks Jobs ([AWS](https://docs.databricks.com/jobs.html)|[Azure](https://docs.microsoft.com/azure/databricks/jobs)|[GCP](https://docs.gcp.databricks.com/jobs.html)).

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Time Travel to correct your data seamlessly

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY vg_mlops_cicd_db.trip_dropoff_features

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM vg_mlops_cicd_db.trip_dropoff_features VERSION AS OF 2

# COMMAND ----------

# MAGIC %md
# MAGIC %sql
# MAGIC <p>
# MAGIC   <span>
# MAGIC RESTORE TABLE VERSION AS OF 2
# MAGIC   </span>
# MAGIC </p>
