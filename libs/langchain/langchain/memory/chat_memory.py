import json
from abc import ABC
from typing import Any, Dict, Optional, Tuple

from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory.utils import get_prompt_input_key
from langchain.pydantic_v1 import Field
from langchain.schema import BaseChatMessageHistory, BaseMemory


class BaseChatMemory(BaseMemory, ABC):
    """Abstract base class for chat memory."""

    chat_memory: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    return_messages: bool = False

    def _get_input_output(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> Tuple[str, str]:
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()

    def toJSON(self) -> str:
        # Use the toJSON method from chat_memory to get string representation
        chat_memory_json = self.chat_memory.toJSON()

        chat_memory_dict = json.loads(chat_memory_json)

        self_dict = {
            "chat_memory": chat_memory_dict
        }

        attributes_to_exclude = ["chat_memory"]

        self_dict.update({key: value for key, value in vars(self).items() if key not in attributes_to_exclude})

        return json.dumps(self_dict, default = lambda o: o.__dict__ , sort_keys=True, indent=4)
