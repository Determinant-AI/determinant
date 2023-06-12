# set up Application Default Credentials, see https://cloud.google.com/docs/authentication/external/set-up-adc for more information

from google.cloud import bigquery

client = bigquery.Client(project="bigquery-public-data")

datasets = client.list_datasets()

descriptions = []

for dataset in datasets:

    dataset_ref = client.get_dataset(dataset.dataset_id)
    description = dataset_ref.description
    tables = list(client.list_tables(dataset))  # Make an API request(s).
    for table in tables:
        table_id = table.table_id
        full_id =  f"{client.project}.{dataset.dataset_id}.{table_id}"

        table = client.get_table(full_id)  # Make an API request.

        column_names = [field.name for field in table.schema]
        print(column_names)
        print("Table description: {}".format(table.description))
        print("Table has {} rows".format(table.num_rows))

    if description:
        descriptions.append((dataset.dataset_id, description))


if __name__ == "__main__":
    pass
