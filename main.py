import sys
import logging
import os.path

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

logging.info("Setting up the OllamaEmbedding")
Settings.embed_model = OllamaEmbedding(
    model_name="mxbai-embed-large",
    base_url="http://localhost:11434",
)

llm = Ollama(model="phi3:mini", request_timeout=360.0)
Settings.llm = llm
Settings.chunk_size = 512

PERSIST_DIR = "./storage"

if not os.path.exists(os.path.join(PERSIST_DIR, "docstore.json")):
    # load the documents and create the index
    documents = SimpleDirectoryReader("./data").load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(llm=llm, streaming=True)
logging.info("Querying...")
response = query_engine.query("What did the author do growing up?")
print("=====================================")
response.print_response_stream()
print("\n====================================")
