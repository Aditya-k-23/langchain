# test_calculator.py
# import json
import pytest
import sys

from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS

EXPECTED_JSON = {
    'lc': 1,
    'type': 'constructor',
    'id': ['langchain', 'memory', 'buffer_window', 'ConversationBufferWindowMemory'],
    'kwargs': {'k': 1},
    'obj': {
        'ai_prefix': 'AI',
        'chat_memory': {
            'id': ['langchain', 'memory', 'chat_message_histories', 'in_memory',
                   'ChatMessageHistory'],
            'kwargs': {},
            'lc': 1,
            'obj': {
                'messages': [
                    {'data': {'additional_kwargs': {},
                              'content': 'hi', 'example': False, 'type': 'human'},
                     'type': 'human'},
                    {'data': {'additional_kwargs': {},
                              'content': "what's up", 'example': False, 'type': 'ai'},
                     'type': 'ai'},
                    {'data': {'additional_kwargs': {},
                              'content': 'not much you', 'example': False, 'type': 'human'},
                     'type': 'human'},
                    {'data': {'additional_kwargs': {},
                              'content': 'not much', 'example': False, 'type': 'ai'},
                     'type': 'ai'}
                ]
            },
            'type': 'constructor'
        },
        'human_prefix': 'Human',
        'input_key': None,
        'k': 1,
        'memory_key': 'history',
        'output_key': None,
        'return_messages': False
    }
}

@pytest.fixture()
def example_memory():
    memory = ConversationBufferWindowMemory(k=1)
    memory.save_context({"input": "hi"}, {"output": "what's up"})
    memory.save_context({"input": "not much you"}, {"output": "not much"})
    memory.load_memory_variables({})
    return memory

def test_to_json(example_memory: ConversationBufferWindowMemory) -> None:
    assert example_memory.to_json()== EXPECTED_JSON

