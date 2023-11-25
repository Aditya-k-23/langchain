import pytest

from langchain.memory import ConversationBufferMemory

EXPECTED_SERIALIZED_MEMORY = "{'lc': 1, 'type': 'constructor', 'id': ['langchain', 'memory', 'buffer', 'ConversationBufferMemory'], 'kwargs': {}, 'obj': {'ai_prefix': 'AI', 'chat_memory': {'id': ['langchain', 'memory', 'chat_message_histories', 'in_memory', 'ChatMessageHistory'], 'kwargs': {}, 'lc': 1, 'obj': {'messages': [{'data': {'additional_kwargs': {}, 'content': 'hi', 'example': False, 'type': 'human'}, 'type': 'human'}, {'data': {'additional_kwargs': {}, 'content': 'what up', 'example': False, 'type': 'ai'}, 'type': 'ai'}]}, 'type': 'constructor'}, 'human_prefix': 'Human', 'input_key': None, 'memory_key': 'history', 'output_key': None, 'return_messages': False}}"

@pytest.fixture()
def example_memory():
    memory = ConversationBufferMemory()
    memory.save_context({'input': 'hi'}, {'output': 'what up'})
    memory.load_memory_variables({})
    return memory

def test_conversion_to_json(example_memory):
    assert str(example_memory.to_json()) == EXPECTED_SERIALIZED_MEMORY