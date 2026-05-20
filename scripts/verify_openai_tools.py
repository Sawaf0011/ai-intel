"""Verify current OpenAI tool-calling API shape."""
import importlib.metadata
import inspect
import openai

print("openai version:", importlib.metadata.version("openai"))

# Check AsyncOpenAI.chat.completions.create signature
client = openai.AsyncOpenAI.__new__(openai.AsyncOpenAI)
# Inspect what parameters chat.completions.create accepts
from openai.resources.chat.completions import AsyncCompletions
sig = inspect.signature(AsyncCompletions.create)
params = list(sig.parameters.keys())
print("chat.completions.create params:", params[:20])

# Verify the tool-call response structure
# Check ChatCompletionMessage attributes
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
print("\nChatCompletionMessage fields:", [f for f in ChatCompletionMessage.model_fields])
print("ChatCompletionMessageToolCall fields:", [f for f in ChatCompletionMessageToolCall.model_fields])
print("Function fields:", [f for f in Function.model_fields])

# Verify the tool result message shape
from openai.types.chat import ChatCompletionToolMessageParam
print("\nTool result message fields:", list(ChatCompletionToolMessageParam.__annotations__))
