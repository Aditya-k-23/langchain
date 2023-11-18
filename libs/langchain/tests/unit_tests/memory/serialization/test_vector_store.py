import faiss

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS

embedding_size_hug = 768

expected = {
    'lc': 1,
    'type': 'constructor',
    'id': [
        'langchain', 'memory', 'vectorstore', 'VectorStoreRetrieverMemory'
    ],
    'kwargs': {},
    'obj': {
        'exclude_input_keys': [],
        'input_key': None,
        'memory_key': 'history',
        'return_docs': False
    }
}

def test_to_json() -> None:
    index = faiss.IndexFlatL2(embedding_size_hug)
    embeddings_hug = HuggingFaceEmbeddings()
    vectorstore = FAISS(embedding_function = embeddings_hug, index = index,
                    docstore = InMemoryDocstore({}), index_to_docstore_id = {})
    retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    memory.save_context({"input": "My favorite food is pizza"},
                        {"output": "that's good to know"})
    memory.save_context({"input": "My favorite sport is soccer"}, {"output": "..."})
    memory.save_context({"input": "I don't the Celtics"}, {"output": "ok"})
    answer = memory.to_json()
    if 'obj' in answer and 'retriever' in answer['obj']:
        del answer['obj']['retriever']
    assert answer == expected