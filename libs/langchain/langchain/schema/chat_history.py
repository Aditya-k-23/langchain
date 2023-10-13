from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from langchain.schema.messages import AIMessage, BaseMessage, HumanMessage
import json

class BaseChatMessageHistory(ABC):
    """Abstract base class for storing chat message history.

    See `ChatMessageHistory` for default implementation.

    Example:
        .. code-block:: python

            class FileChatMessageHistory(BaseChatMessageHistory):
                storage_path:  str
                session_id: str

               @property
               def messages(self):
                   with open(os.path.join(storage_path, session_id), 'r:utf-8') as f:
                       messages = json.loads(f.read())
                    return messages_from_dict(messages)

               def add_message(self, message: BaseMessage) -> None:
                   messages = self.messages.append(_message_to_dict(message))
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       json.dump(f, messages)

               def clear(self):
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       f.write("[]")
    """

    messages: List[BaseMessage]
    """A list of Messages stored in-memory."""

    def add_user_message(self, message: str) -> None:
        """Convenience method for adding a human message string to the store.

        Args:
            message: The string contents of a human message.
        """
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Convenience method for adding an AI message string to the store.

        Args:
            message: The string contents of an AI message.
        """
        self.add_message(AIMessage(content=message))

    @abstractmethod
    def add_message(self, message: BaseMessage) -> None:
        """Add a Message object to the store.

        Args:
            message: A BaseMessage object to store.
        """
        raise NotImplementedError()

    @abstractmethod
    def clear(self) -> None:
        """Remove all messages from the store"""

    def toJSON(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)

    @classmethod
    def fromJSON(cls, json_input: str) -> None:
        memory_dict = json.loads(json_input)

        # Create a mapping from type names to message classes
        message_type_mapping = {
            'human': HumanMessage,
            'ai': AIMessage
        }

        # Convert JSON data to message objects with the correct type
        messages = [message_type_mapping[msg['type']](**msg) for msg in memory_dict['messages']]

        # Extract additional attributes from memory_dict
        additional_attributes = {key: memory_dict[key] for key in memory_dict if key != 'messages'}

        # Use **kwargs to pass both messages and additional attributes to cls
        return cls(messages=messages, **additional_attributes)

