import os
import uuid
import json
import datetime
from time import time, sleep
import numpy as np

import pinecone
import openai

openai.api_key=os.getenv("OPENAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east1-gcp")
pinecone_index = pinecone.Index("brienne-mvp")

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)

def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")

def pinecone_upcert(uuid_str, addr):
    vectors = []
    msg_dict = {}
    
    #uuid_str = str(uuid.uuid4().hex)[:512]
    msg_dict["id"] = uuid_str
    msg_dict["values"] = addr

    vectors.append(msg_dict)

    resp = pinecone_index.upsert(
        vectors=vectors,
        namespace="chatlogs"
    )

def embed_message(message, engine="text-embedding-ada-002"):
    # Generate an embedding using the OpenAI GPT-3 model
    response = openai.Embedding.create(
        input=message,
        model=engine,
        max_tokens=1024,
        n=0,
        stop=None,
        temperature=0,
    )

    embedding = response['data'][0]['embedding']
    
    # Convert the embedding to a numpy array and normalize it to unit length
    normalized_embedding = np.array(embedding) / np.linalg.norm(embedding)
    return normalized_embedding.tolist()

###
# remember a user_input into episodic memory
def remember(semantic_addr, timestamp, uuid_str, speaker, input):
    # store semantic address onto pinecone
    pinecone_upcert(uuid_str, semantic_addr)
    # save locally
    json_str = {
        'speaker': speaker, 
        'time': timestamp,
        'message': input,
        'uuid': uuid_str,
        'sem_addr': semantic_addr}
    file_name = 'log_%s' % uuid_str;
    save_json('data/epi/%s' % file_name, json_str);
    
#####################################Main###############################

# init
if not os.path.exists("data/epi"):
    os.makedirs("data/epi")


# main loop
while True:
    # Accept input from user
    raw_input = input(">>> ")
    timestamp = time()
    timestamp_str = timestamp_to_datetime(timestamp)
    
    # get  semantic address
    semantic_addr = embed_message(raw_input)
    
    # format user message
    fmt_input = '%s: %s - %s' % ('USER', timestamp_str, raw_input)
    
    # remember this input
    uuid_str = str(uuid.uuid4().hex)[:512]
    remember(semantic_addr, timestamp, uuid_str, 'USER', raw_input)

    print('s')    
    # Step 4: Fetch memory threads from Pinecone according to recent 30 messages
    #search_results = pinecone.query(index_name="brienne-mvp", data=embedding, 
    #                                top_k=30, include_ids=True)
    #if len(search_results.ids) > 1:
    #    print("Recent memory threads:")
    #    for id in search_results.ids[1:]:
    #        print(id.decode("utf-8"))

    

