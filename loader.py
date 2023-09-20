from dotenv import dotenv_values

import csv
import redis
import json
import openai

from redis.commands.search.field import TextField, NumericField, TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

import pandas as pd


config = dotenv_values('.env')
redisHost = config['REDIS_HOST']
redisUsername = config['REDIS_USERNAME']
redisPassword = config['REDIS_PASSWORD']

openai.api_key = config['OPENAI_API_KEY']

client = redis.Redis(
    host=redisHost,
    port=19920,
    username=redisUsername,
    password=redisPassword)

client.flushall()

df = pd.read_csv('resources/sets.csv')
json_str = df.to_json(orient='records')
sets = json.loads(json_str)

pipeline = client.pipeline()
for i, current_set in enumerate(sets, start=1):
    redis_key = f"sets:{i:09}"
    #add an embedding field to the current_set
    current_set["embedding"] = openai.Embedding.create(input=json.dumps(current_set), model="text-embedding-ada-002")['data'][0]['embedding']
    print(json.dumps(current_set))
    pipeline.json().set(redis_key, "$", current_set)

res = pipeline.execute()

schema = (
    TextField("$.set_num", no_stem=True, as_name="set_num"),
    TextField("$.name", no_stem=True, as_name="name"),
    NumericField("$.year", as_name="year"),
    NumericField("$.theme_id", as_name="theme"),
    NumericField("$.num_parts", as_name="num_parts"),
    VectorField(
        "$.description_embeddings",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": 1536,
            "DISTANCE_METRIC": "COSINE",
        },
        as_name="vector",
    ),
)
definition = IndexDefinition(prefix=["sets:"], index_type=IndexType.JSON)
res = client.ft("idx:sets_vss").create_index(
    fields=schema, definition=definition
)

# following https://redis.io/docs/interact/search-and-query/search/vectors/