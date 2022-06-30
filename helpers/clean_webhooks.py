# Databricks notebook source
import mlflow
from mlflow.utils.rest_utils import http_request
import json

def client():
  return mlflow.tracking.client.MlflowClient()

host_creds = client()._tracking_client.store.get_host_creds()
host = host_creds.host
token = host_creds.token

def mlflow_call_endpoint(endpoint, method, body='{}'):
  if method == 'GET':
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body))
  else:
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, json=json.loads(body))
  return response.json()

# COMMAND ----------

model_name = 'vg_mlops_cicd_nyc_taxi_fare_model'

list_model_webhooks = json.dumps({"model_name": model_name})

webhook_dict = mlflow_call_endpoint("registry-webhooks/list", method = "GET", body = list_model_webhooks)

webhook_dict

# COMMAND ----------

for webhook in webhook_dict.get('webhooks'):
  id = str(webhook.get('id'))
  print('deleting webhook on model '+model_name+' : '+id)
  mlflow_call_endpoint("registry-webhooks/delete",
                     method="DELETE",
                     body = json.dumps({'id': id}))
