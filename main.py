import numpy as np
from dotenv import dotenv_values

import openai

import redis

from redis.commands.search.query import Query

config = dotenv_values('.env')
redisHost = config['REDIS_HOST']
redisUsername = config['REDIS_USERNAME']
redisPassword = config['REDIS_PASSWORD']

client = redis.Redis(
    host=redisHost,
    port=19920,
    username=redisUsername,
    password=redisPassword)

# Generate embedding to request
openai.api_key = config['OPENAI_API_KEY']

info = client.ft("idx:sets_vss").info()

embedding = openai.Embedding.create(input="ninja", model="text-embedding-ada-002")['data'][0]['embedding']

query = (
    Query('(*)=>[KNN 3 @vector $query_vector AS vector_score]')
    .sort_by('vector_score')
    .return_fields('set_num', 'name', 'year', 'theme_id', 'num_parts')
    .dialect(2)
)

result = client.ft("idx:sets_vss").search(query, {'query_vector': np.array(embedding, dtype=np.float32).tobytes()}).docs

print(result)
