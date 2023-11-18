import getpass
import os

import faiss

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import FakeEmbeddings, HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.memory import (
    ChatMessageHistory,
    ConversationBufferWindowMemory,
    VectorStoreRetrieverMemory,
)
from langchain.vectorstores import FAISS

history = ChatMessageHistory(input_key="your_input_key_value")
history.add_user_message("hi!")

history.add_ai_message("what's up?")

history.add_ai_message("what's up 2?")

print("Current Messages Stored in history: ")
print(history.messages)

print("Serialized history:")
print(history.to_json())

memory = ConversationBufferWindowMemory(k = 1)
memory.save_context({"input": "hi"}, {"output": "what's up"})
memory.save_context({"input": "not much you"}, {"output": "not much"})
memory.load_memory_variables({})
print("Serialized Memory: ")
print(memory.to_json())


print("*****************Debug Area****************")
embedding_size = 1536 # Dimensions of the OpenAI and Fake embeddings
embedding_size_hug = 768 # Dimensions of the HuggingFaceEmbeddings

# Pass correct embedding size depending on which
index = faiss.IndexFlatL2(embedding_size_hug)

# Use key Aditya Put in discord
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAiKey: ")

# Code to try setting up langsmith
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - 33"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
#os.environ["LANGCHAIN_API_KEY"] = "<YOUR-API-KEY>"


embeddings = FakeEmbeddings(size=1536)

embeddings_hug = HuggingFaceEmbeddings()

embeddings_open = OpenAIEmbeddings().embed_query

# Two different methods of initializing FAISS, make to pass relevant embedding
vectorstore = FAISS(embedding_function = embeddings_hug, index = index,
                    docstore = InMemoryDocstore({}), index_to_docstore_id = {})
#vectorstore = FAISS.from_texts(["waow"], embeddings_open)

# In actual usage, you would set `k` to be a higher value, but we use k=1 to show that
# the vector lookup still returns the semantically relevant information
retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)

# When added to an agent, the memory object can save pertinent information
# from conversations or used tools
memory.save_context({"input": "My favorite food is pizza"}, {"output": "that's good to know"})
memory.save_context({"input": "My favorite sport is soccer"}, {"output": "..."})
memory.save_context({"input": "I don't the Celtics"}, {"output": "ok"})
print("Searialized VectoreStoreRetrieverMemory: ")
print(memory.to_json())

print("||||||||||||||||||||||||||||")
# Notice the first result returned is the memory pertaining to tax help, which the language
# model deems more semantically relevant to a 1099 than the other documents, despite them both containing numbers.
print(memory.load_memory_variables({"prompt": "what sport should i watch?"})["history"])
print("||||||||||||||||||||||||||||")
print(memory.to_json())
print('finished')