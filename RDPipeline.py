# Libraries: pyspark, pandas, matplotlib, requests, azure-storage-blob, azure-keyvault-secrets, azure-identity
# Install with: pip install pyspark azure-storage-blob pandas matplotlib requests azure-keyvault-secrets azure-identity

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
import pandas as pd
import requests
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

class RDPipeline:
    def __init__(self, csv_data_path, azure_connection_string, azure_container, azure_blob_name):
        self.spark = SparkSession.builder.appName("Risk Insights Platform").getOrCreate()
        self.csv_data_path = csv_data_path
        self.azure_connection_string = azure_connection_string
        self.azure_container = azure_container
        self.azure_blob_name = azure_blob_name

    def fetch_external_data(self):
        """
        Fetches data from a real (open source) weather-data API.
        """
        api_url = "https://api.open-meteo.com/v1/forecast"
        regions = ["North America", "Europe", "Asia"]
        risk_types = ["Flood", "Storm", "Earthquake"]

        api_data = []

        for region, risk_type in zip(regions, risk_types):
            response = requests.get(
                api_url,
                params={"latitude": 40.7128, "longitude": -74.0060, "hourly": "temperature_2m"}
            )

            if response.status_code == 200:
                risk_score = response.json().get("hourly", {}).get("temperature_2m", [0])[0]
                api_data.append({"region": region, "risk_type": risk_type, "risk_score": risk_score})
            else:
                print(f"Failed to fetch data for {region} - {risk_type}: {response.status_code}")

        return pd.DataFrame(api_data)

    def load_csv_data(self):
        """
        Load data from a csv file.
        """
        return self.spark.read.csv(self.csv_data_path, header=True, inferSchema=True)

    def process_data(self, api_df, csv_df):
        api_spark_df = self.spark.createDataFrame(api_df)
        processed_data = (
            csv_df
            .union(api_spark_df)
            .groupBy("region", "risk_type")
            .agg(avg("risk_score").alias("avg_risk_score"))
        )
        return processed_data

    def upload_to_azure_blob(self, processed_data):
        """
        Upload (processed) data to Azure Blob Storage.
        """
        blob_service_client = BlobServiceClient.from_connection_string(self.azure_connection_string)
        container_client = blob_service_client.get_container_client(self.azure_container)

        # Convert Spark DataFrame to Pandas DataFrame for upload
        pandas_df = processed_data.toPandas()
        csv_data = pandas_df.to_csv(index=False)

        blob_client = container_client.get_blob_client(self.azure_blob_name)
        blob_client.upload_blob(csv_data, overwrite=True)

    def visualize_data(self, processed_data):
        """
        Visualize the average risk score by region.
        """
        pandas_df = processed_data.toPandas()
        pandas_df.groupby("region")["avg_risk_score"].mean().plot(kind="bar", title="Average Risk Score by Region")

    def run_pipeline(self):
        # Simulate API and File Data
        api_data = self.fetch_external_data()
        csv_data = self.load_csv_data()

        processed_data = self.process_data(api_data, csv_data)

        self.upload_to_azure_blob(processed_data)

        self.visualize_data(processed_data)

# Main Execution
if __name__ == "__main__":
    csv_data_path = "risk_data.csv"  # Replace with the path to your CSV file
    azure_connection_string = "<Your Azure Connection String>"
    azure_container = "risk-data"
    azure_blob_name = "processed_risk_data.csv"

    pipeline = RiskInsightsPipeline(csv_data_path, azure_connection_string, azure_container, azure_blob_name)
    pipeline.run_pipeline()
