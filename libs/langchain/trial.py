from langchain.memory import (
    ChatMessageHistory,
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
)

history = ChatMessageHistory(input_key="your_input_key_value")
history.add_user_message("hi!")

history.add_ai_message("what's up?")

history.add_ai_message("what's up 2?")

print("Current Messages Stored in history: ")
print(history.messages)

print("Serialized history:")
print(history.toJSON())

history_deserialized = ChatMessageHistory.fromJSON(history.toJSON())

print("Original history: ")
print(history)
print("Deserialized history: ")
print(history_deserialized)

memory = ConversationBufferMemory()
memory.save_context({"input": "hi"}, {"output": "what's up"})
memory.load_memory_variables({})
print("Serialized memory: ")
print(memory.toJSON())

memory_window = ConversationBufferWindowMemory( k=1, return_messages=True)
memory_window.save_context({"input": "hi"}, {"output": "sup buddy"})
memory_window.save_context({"input": "not much you"}, {"output": "not much"})
print("Serialized memory_window: ")
print(memory_window.toJSON())