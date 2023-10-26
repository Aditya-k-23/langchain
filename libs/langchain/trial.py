from langchain.memory import ChatMessageHistory, ConversationBufferMemory

history = ChatMessageHistory(input_key="your_input_key_value")
history.add_user_message("hi!")

history.add_ai_message("what's up?")

history.add_ai_message("what's up 2?")

print("Current Messages Stored in history: ")
print(history.messages)

print("Serialized history:")
print(history.to_json())

memory = ConversationBufferMemory()
memory.save_context({"input": "hi"}, {"output": "what's up"})
memory.load_memory_variables({})
print("Serialized Memory: ")
print(memory.to_json())
