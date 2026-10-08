import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Access the client
_client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    #stream=True
    )

def get_LLM_response(prompt: str, memory=None):
    """Sends a prompt to DeepSeek V4.1 Flash and returns (reply, updated_memory)"""

    # Add prompt to the LLM memory
    if not memory:
        print("Memory is empty, starting a new conversation")
        memory = []
        memory.append({"role": "system", "content": "You are a concise assistant."})

    memory.append({"role": "user", "content": prompt})

    # Send a call to an LLM with the memory
    response = _client.chat.completions.create(
        model="deepseek-flash",
        messages=memory,
        #reasoning_effort="high/none/low",
        #extra_body={"thinking": {"type": "enabled"}}
    )

    # Add the LLM's response to the memory
    msg = response.choices[0].message
    memory.append({"role": "assistant", "content": msg.content})

    return msg.content, memory

response1, memory = get_LLM_response("Write a sentence that evokes happiness")
response2, memory = get_LLM_response("Write a sentence that evokes sadness", memory)

print(response1)
print(response2)
