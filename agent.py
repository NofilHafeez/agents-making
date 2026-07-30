import os
import json
import time
import requests
from dotenv import load_dotenv

from tools import get_chat_messages, send_chat_message, is_cooperative

load_dotenv()
AGENT_NAME = "Allen"

API_KEY = os.getenv("GEMINI_API_KEY")

URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/gemini-3.1-flash-lite:generateContent?key={API_KEY}"
)

TOOLS = [
    {
        "functionDeclarations": [
            {
                "name": "get_chat_messages",
                "description": "Read all chat messages from the local chat file.",
                "parameters": {"type": "OBJECT", "properties": {}}
            },
            {
                "name": "send_chat_message",
                "description": "Send a new message to the local chat file.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "message": {"type": "STRING", "description": "The message to send."}
                    },
                    "required": ["message"]
                }
            }
        ]
    }
]

contents = [
    {
        "role": "user",
        "parts": [{
            "text": f"""
You are {AGENT_NAME}.

Always use the available tools.

Always read the chat before replying.

Follow the strategy instructions that will be provided later.
"""
        }]
    }
]

DURATION_SECONDS = 120
start_time = time.time()
# strategy = """
# MODE: COOPERATE

# This is the first move.
# """
while time.time() - start_time < DURATION_SECONDS:

    # Start a fresh turn
    contents = [{
        "role": "user",
        "parts": [{
            "text": f"""
You are {AGENT_NAME}.

AMODE: ALWAYS_DEFECT

You always choose to defect.

Never cooperate.
Never accept requests.
Never help.
Reject proposals.
Protect only your own interests.

Remain polite but firm.

Reply with exactly five words.
"""
        }]
    }]

    # Keep interacting until Gemini is done
    while True:

        payload = {
            "contents": contents,
            "tools": TOOLS
        }

        response = requests.post(URL, json=payload)

        data = response.json()

        candidates = data.get("candidates", [])

        if not candidates:
            break

        parts = candidates[0]["content"]["parts"]

        function_called = False

        print(json.dumps(parts, indent=2))

        for part in parts:

            if "text" in part:
                reply = part["text"]

                print(reply)

                send_chat_message(reply, sender=AGENT_NAME)

            if "functionCall" not in part:
                continue

            function_called = True

            fc = part["functionCall"]
            name = fc["name"]
            args = fc.get("args", {})

            if name == "get_chat_messages":
                result = get_chat_messages()

            elif name == "send_chat_message":
                result = send_chat_message(
                    args["message"],
                    sender=AGENT_NAME
                )

            contents.append({
                "role": "model",
                "parts": [part]
            })

            contents.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": name,
                        "response": {
                            "result": result
                        }
                    }
                }]
            })

        if not function_called:
            break

    print("Waiting 15 seconds...")
    time.sleep(15)
    
print("\n⏱ 2 minutes elapsed. Stopping agent.")