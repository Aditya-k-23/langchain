import json

from langchain.memory import ChatMessageHistory, ConversationBufferMemory

'''
The code below tests out implementation for to_json and from_json in the
BaseChatMessageHistory Class
'''
history = ChatMessageHistory(input_key="your_input_key_value")
history.add_user_message("hi!")

history.add_ai_message("what's up?")

print("Current Messages Stored in history: ")
print(history.messages)

print("Serialized history:")
print(history.to_json())



loaded_history = ChatMessageHistory.from_json(json.dumps(history.to_json()))

print("Deserialized history: ")
print(loaded_history)
print("Original history: ")
print(history)


'''
The code below tests out implementation for to_json and from_json in the
BaseChatMemory Class
'''
print("\n")
memory = ConversationBufferMemory()

memory.save_context({"input": "hi"}, {"output": "what's up"})
memory.load_memory_variables({})

print("Serialized Memory: ")
serialized = memory.to_json()
print(serialized)

loaded_memory = ConversationBufferMemory.from_json(json.dumps(serialized))

print("Deserialized memory:")
print(loaded_memory)
print("Original memory:")
print(memory)
