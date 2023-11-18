# test_calculator.py
# import json
import pytest

from langchain.chains import ConversationChain
from langchain.llms import HuggingFace
from langchain.memory import ConversationBufferWindowMemory

expected_json = {
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
                    {'data': {'additional_kwargs': {}, 'content': 'hi', 'example': False, 'type': 'human'},
                     'type': 'human'},
                    {'data': {'additional_kwargs': {}, 'content': "what's up", 'example': False, 'type': 'ai'},
                     'type': 'ai'},
                    {'data': {'additional_kwargs': {}, 'content': 'not much you', 'example': False, 'type': 'human'},
                     'type': 'human'},
                    {'data': {'additional_kwargs': {}, 'content': 'not much', 'example': False, 'type': 'ai'},
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


def test_to_json() -> None:
    memory = ConversationBufferWindowMemory(k=1)
    memory.save_context({"input": "hi"}, {"output": "what's up"})
    memory.save_context({"input": "not much you"}, {"output": "not much"})
    memory.load_memory_variables({})
    actual_json = memory.to_json()
    assert (actual_json == expected_json)

def test_in_chain() -> None:
    conversation_with_summary = ConversationChain(
    llm = HuggingFace(model_name="sshleifer/tiny-gpt2") ,
    # We set a low k=2, to only keep the last 2 interactions in memory
    memory=ConversationBufferWindowMemory(k=2),
    verbose=True
    )
    conversation_with_summary.predict(input="Hi, what's up?")
