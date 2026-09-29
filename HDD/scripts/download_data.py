from google.cloud import bigquery
import pandas as pd

# Initialize the client using your specific project ID from the console
client = bigquery.Client(project="gen-lang-client-0015191707")

# Define the query to grab 100 million rows
# Query 10 million rows from the actual telemetry table
query = """
    SELECT * 
    FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_usage` 
    LIMIT 10000000
"""

print("Querying and downloading data... This will take a while.")
dataframe = client.query(query).to_dataframe(create_bqstorage_client=True)

# Save as instance_usage instead of machine_events
output_path = r"C:\Users\sabbu\OneDrive\Documents\Git\System-Degradation-Predictor\Data\BORG\instance_usage.csv"
dataframe.to_csv(output_path, index=False)


print(f"Download complete! Saved to: {output_path}")