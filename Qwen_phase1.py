import os
from aiolimiter import AsyncLimiter
import asyncio
import logging
import numpy as np
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

os.environ["OPENAI_API_KEY"]  = "sk-125fa70b20b84d0d9e3208604209c726"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
load_dotenv('.env')
MODEL = "qwen-plus"
WORKING_DIR = "./nano_graphrag_cache_qwen_TEST"
EMBED_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", cache_folder=WORKING_DIR, device="cpu"
)

# We're using Sentence Transformers to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


async def qwen_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages) # Compute hash value
        cached_response = await hashing_kv.get_by_id(args_hash) # Get cached item
        if cached_response is not None:
            return cached_response["return"]

    rate_limiter = AsyncLimiter(max_rate=180, time_period=60)  # Maxinum 180 responses/min
    
    async with rate_limiter: # Limit with rate limiter
        while True:
            try:
                response = await openai_async_client.chat.completions.create(
                    model=MODEL, messages=messages, **kwargs
                )
                
                if hashing_kv is not None:
                    await hashing_kv.upsert(
                        {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
                    )
                
                return response.choices[0].message.content
            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    print(f"Rate limit exceeded. Retrying in 5 seconds...")
                    await asyncio.sleep(5)  # Wait 5 seconds
                else:
                    await asyncio.sleep(0)
                    #pass


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=qwen_model_if_cache,
        cheap_model_func=qwen_model_if_cache,
        embedding_func=local_embedding,
    )

    print(
        rag.query("What are the articles about?", param=QueryParam(mode="global"))
    )



def insert():
    from time import time
    FAKE_TEXT = ""
    for filename in os.listdir("./txtWhitePapers"): # Read all contents in the folder
        with open(os.path.join("./txtWhitePapers", filename), "r") as file:
            FAKE_TEXT += file.read()

    #remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    #remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    #remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    #remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    #remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        chunk_token_size=800,
        enable_llm_cache=True,
        best_model_func=qwen_model_if_cache,
        cheap_model_func=qwen_model_if_cache,
        embedding_func=local_embedding,
    )
    start = time()
    rag.insert(FAKE_TEXT)
    print("indexing time:", time() - start)



if __name__ == "__main__":
    insert()
    query()