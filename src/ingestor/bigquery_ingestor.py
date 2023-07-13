from google.cloud import bigquery
from deeplake.core.vectorstore.deeplake_vectorstore import DeepLakeVectorStore

import json
import tqdm
import os
import random
import openai
from datetime import datetime
import argparse
import time
from tqdm import tqdm

# Set OpenAI API key
#openai.openai.api_key = "sk-xxxx"

# Download BigQuery meta-data(description, column names, number of rows)
client = bigquery.Client(project="bigquery-public-data")
datasets = client.list_datasets()
all_data = []

try:
    for dataset in datasets:

        dataset_ref = client.get_dataset(dataset.dataset_id)
        dataset_description = "" if dataset_ref.description is None else dataset_ref.description
        tables = list(client.list_tables(dataset))  # Make an API request(s).
        for table in tables:

            table_dict = {}
            table_id = table.table_id
            full_id =  f"{client.project}.{dataset.dataset_id}.{table_id}"

            table_ref = client.get_table(full_id)  # Make an API request.
            table_description = "" if table_ref.description is None else table_ref.description
            column_names = [field.name for field in table_ref.schema]
            table_dict['full_id'] = full_id
            table_dict['columns'] = column_names
            table_dict['description'] = dataset_description + "\n" + table_description
            table_dict['num_rows'] = table_ref.num_rows

            print(full_id)
            all_data.append(table_dict)

except bigquery.ClientError as e:
    print(f"BigQuery Client error occurred: {str(e)}")
except Exception as e:
    print(f"An error occurred: {str(e)}")

# Filter out data with bad descriptions
filtered_all_data = [d for d in all_data if len(d['description']) >= 10 and len(d['columns']) > 1]

with open("all_bigquery_tables.jsonl", "w") as outfile:
    for entry in filtered_all_data:
        json.dump(entry, outfile)
        outfile.write('\n')

# Use OpenAI embedding function for descriptions
def embedding_function(texts, model="text-embedding-ada-002"):
   
   if isinstance(texts, str):
       texts = [texts]

   texts = [t.replace("\n", " ") for t in texts]
   return [data['embedding']for data in openai.Embedding.create(input = texts, model=model)['data']]


# Use ActiveLoop for vector store
chunked_text = [d['description'] for d in filtered_all_data]
keys_to_use = ['full_id', 'columns', 'num_rows']
metadata = [{k: d[k] for k in keys_to_use} for d in filtered_all_data]

vector_store_path = 'bigquery_meta_vector_store'
vector_store = DeepLakeVectorStore(
    path = vector_store_path,
)
vector_store.add(text = chunked_text, 
                 embedding_function = embedding_function, 
                 embedding_data = chunked_text, 
                 metadata = metadata
)


def make_requests(
        engine, prompts, max_tokens, temperature, top_p, 
        frequency_penalty, presence_penalty, stop_sequences, logprobs, n, best_of, retries=3, api_key=None, organization=None
    ):
    response = None
    target_length = max_tokens
    if api_key is not None:
        openai.api_key = api_key
    if organization is not None:
        openai.organization = organization
    retry_cnt = 0
    backoff_time = 30
    while retry_cnt <= retries:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompts,
                max_tokens=target_length,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop_sequences,
                logprobs=logprobs,
                n=n,
                best_of=best_of,
            )
            break
        except openai.error.OpenAIError as e:
            print(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                target_length = int(target_length * 0.8)
                print(f"Reducing target length to {target_length}, retrying...")
            else:
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5
            retry_cnt += 1
    
    if isinstance(prompts, list):
        results = []
        for j, prompt in enumerate(prompts):
            data = {
                "prompt": prompt,
                "response": {"choices": response["choices"][j * n: (j + 1) * n]} if response else None,
                "created_at": str(datetime.now()),
            }
            results.append(data)
        return results
    else:
        data = {
            "prompt": prompts,
            "response": response,
            "created_at": str(datetime.now()),
        }
        return [data]


# Generate self-instruct data for fine-tuning
template = """I have a table in bigquery:
{table_metadata}
Generate a pair of relevant question and corresponding SQL code, make sure the generated pairs is diversified and cover different use cases.
Return a list of json object keyed by "question" and "sql" for your generation, use escape double quotes(\") in SQL code
"""

output_sql_qa = []
# num_batches = len(filtered_all_data)
num_batches = len(filtered_all_data)
for i in tqdm(range(num_batches)):
    prompt = template.format(table_metadata=str(filtered_all_data[i]), num_q=1)
    response = openai.Completion.create(
        engine='text-davinci-003',  # Replace with the appropriate GPT-4 engine name
        prompt=prompt,
        temperature=1,
        top_p=1,
        n=5,
        max_tokens=4097 - len(prompt)  # Replace with the desired maximum number of tokens for the completion
    )
    text = response.choices[0].text

    # Post-processing of generated texts
    # Split the text into lines
    # lines = text.strip().split('\n')
    output = None
    print(text)
    try:
        output = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    if output is not None:
        _ = [o.update({'context': filtered_all_data[i]}) for o in output]
        output_sql_qa.extend(output)


with open("output_sql_qa.jsonl", "w") as outfile:
    for entry in filtered_all_data:
        json.dump(entry, outfile)
        outfile.write('\n')



"""
<OpenAIObject text_completion id=cmpl-7Y1vCsUFUZgyWRKCHzLMt3v1wzTqp at 0x1077a35e0> JSON: {
  "id": "cmpl-7Y1vCsUFUZgyWRKCHzLMt3v1wzTqp",
  "object": "text_completion",
  "created": 1688344894,
  "model": "text-davinci-003",
  "choices": [
    {
      "text": "\nQuestion: \nWhat is the maximum value and upper confidence interval for the \"Leading Cause of Death\" measure in New York in 2017?\n\nSQL Code:\nSELECT MAX(value), upper_ci \nFROM `bigquery-public-data.america_health_rankings.ahr`\nWHERE edition = 2017\n  AND measure_name = 'Leading Cause of Death'\n  AND state_name = 'New York'\n;",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 126,
    "completion_tokens": 99,
    "total_tokens": 225
  }
}
"""


# Initialize variables
pairs = []
current_question = ''
current_query = ''

# Process each line
for line in lines:
    line = line.strip()
    if line.startswith('SELECT'):
        # Store the previous question and query pair
        if current_question and current_query:
            pairs.append((current_question, current_query))
        # Reset variables for the new question and query
        current_question = line
        current_query = ''
    else:
        current_query += line + ' '

# Store the last question and query pair
if current_question and current_query:
    pairs.append((current_question, current_query))

# Display the pairs
for question, query in pairs:
    print("Question:", question)
    print("SQL Query:", query)


if __name__ == "__main__":
    prompt = "how many MLB games have been player last year?"
    search_results = vector_store.search(embedding_data=prompt, embedding_function=embedding_function)
    search_results['text'][0]
    search_results['metadata'][0]

# 'Overview: This public data includes pitch-by-pitch data for Major League Baseball (MLB) games in 2016. This dataset contains the following tables: games_wide (every pitch, steal, or lineup event for each at bat in the 2016 regular season), games_post_wide(every pitch, steal, or lineup event for each at-bat in the 2016 post season), and schedules ( the schedule for every team in the regular season). The schemas for the games_wide and games_post_wide tables are identical. With this data you can effectively replay a game and rebuild basic statistics for players and teams.\n\nUpdate frequency: Historic (none)\n\nDataset source: SportRadar\n\nTerms of use: Copyright Sportradar LLC. Access to data is intended solely for internal research and testing purposes, and is not to be used for any business or commercial purpose. Data are not to be exploited in any manner without express approval from Sportradar. Display of data must include the phrase, “Data provided by Sportradar LLC,” and be hyperlinked to\xa0www.sportradar.com\n\nSee the GCP Marketplace listing for more details and sample queries: https://console.cloud.google.com/marketplace/details/sportradar-public-data/mlb-pitch-by-pitch\n

# {'full_id': 'bigquery-public-data.baseball.schedules',
#  'columns': ['gameId',
#   'gameNumber',
#   'seasonId',
#   'year',
#   'type',
#   'dayNight',
#   'duration',
#   'duration_minutes',
#   'homeTeamId',
#   'homeTeamName',
#   'awayTeamId',
#   'awayTeamName',
#   'startTime',
#   'attendance',
#   'status',
#   'created'],
#  'num_rows': 2431}
