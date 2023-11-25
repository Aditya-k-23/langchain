import pytest

from langchain.memory import ConversationTokenBufferMemory
from tests.unit_tests.llms.fake_llm import FakeLLM

EXPECTED_SERIALIZED_MEMORY = "{'lc': 1, 'type': 'constructor', 'id': ['langchain', 'memory', 'token_buffer', 'ConversationTokenBufferMemory'], 'kwargs': {'llm': FakeLLM(), 'max_token_limit': 100}, 'obj': {'ai_prefix': 'AI', 'chat_memory': {'id': ['langchain', 'memory', 'chat_message_histories', 'in_memory', 'ChatMessageHistory'], 'kwargs': {}, 'lc': 1, 'obj': {'messages': [{'data': {'additional_kwargs': {}, 'content': 'hi', 'example': False, 'type': 'human'}, 'type': 'human'}, {'data': {'additional_kwargs': {}, 'content': 'whats up', 'example': False, 'type': 'ai'}, 'type': 'ai'}]}, 'type': 'constructor'}, 'human_prefix': 'Human', 'input_key': None, 'llm': {'id': ['tests', 'unit_tests', 'llms', 'fake_llm', 'FakeLLM'], 'lc': 1, 'repr': 'FakeLLM()', 'type': 'not_implemented'}, 'max_token_limit': 100, 'memory_key': 'history', 'output_key': None, 'return_messages': False}}"

@pytest.fixture()
def example_memory():
    memory = ConversationTokenBufferMemory(llm=FakeLLM(), max_token_limit=100)
    memory.save_context({'input': 'hi'}, {'output': 'whats up'})
    memory.load_memory_variables({})
    return memory

def test_conversion_to_json(example_memory: ConversationTokenBufferMemory):
    assert str(example_memory.to_json()) == EXPECTED_SERIALIZED_MEMORY
