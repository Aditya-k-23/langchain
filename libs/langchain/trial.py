from langchain.memory import ChatMessageHistory, PostgresChatMessageHistory

history = ChatMessageHistory(input_key="your_input_key_value")
history.add_user_message("hi!")

history.add_ai_message("what's up?")

history.add_ai_message("what's up 2?")

print(history.messages)

print(history.toJSON())

history_deserialized = ChatMessageHistory.fromJSON(history.toJSON())

print(history)
print(history_deserialized)