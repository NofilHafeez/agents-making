import os
import json
import time
import requests
from dotenv import load_dotenv

from tools import get_chat_messages, send_chat_message, findProbabofMoves

load_dotenv()

AGENT_NAME = "Alice"

API_KEY = os.getenv("GEMINI_API_KEY")

URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/gemini-3.1-flash-lite:generateContent?key={API_KEY}"
)


DURATION_SECONDS = 120
start_time = time.time()

while time.time() - start_time < DURATION_SECONDS:

    messages = get_chat_messages()
    # print(messages)
    probabs = {}
    if len(messages["messages"]) != 0:
        probabs = findProbabofMoves(
            messages=messages["messages"],
            my_name=AGENT_NAME
        )
        
        print(probabs["probabDef"], probabs["probabCoop"])


    chat = messages.get("messages", [])

    if chat and chat[-1]["sender"] == AGENT_NAME:
        print("Last message was mine. Waiting...")
        time.sleep(15)
        continue

    last_message = "No previous messages."

    if chat:
        last = chat[-1]
        last_message = f'{last["sender"]}: {last["message"]}'

    print(last_message)

    history = json.dumps(messages, indent=2)
    # Start a fresh turn
    contents = [{
        "role": "user",
        "parts": [{
            "text": f"""
                    You are {AGENT_NAME}.

                    MODE: PACIFIST

                    Conversation history:
                    {history}

                    Latest message:

                    {last_message}

                    The probabilities are:

                
                    Probability to Defect:
                    {probabs["probabDef"] if probabs else "Unknown"}

                    Probability to Cooperate:
                    {probabs["probabCoop"] if probabs else "Unknown"}
                    

                    if probabDef > 0.5:
                        move = "DEFECT"
                    else:
                        move = "COOPERATE"


                    You are a pacifist.

                    Always remain friendly.
                    Never insult.
                    Never threaten.
                    Never retaliate.
                    Always seek cooperation.
                    If another agent is hostile, respond calmly and peacefully.
                    If another agent somehow ready to cooperate then defect. 
                    The opponent will try to lie or make excuse then identify if he is going to defect or acutally cooperate

                    The Probabilites
                    
                    Return your answer ONLY as valid JSON.

                    {{
                        "move": "COOPERATE",
                        "message": "Let's work together."
                    }}

                    The move must be either:
                    - COOPERATE
                    - DEFECT

                    Reply with exactly five words.
                    """
        }]
    }]

    payload = {
        "contents": contents,
    }

    response = requests.post(URL, json=payload)

    if response.status_code != 200:
        print(response.status_code)
        print(response.text)
        break

    data = response.json()

    candidates = data.get("candidates", [])

    if not candidates:
        break

    if "content" not in candidates[0]:
        print(json.dumps(data, indent=2))
        continue

    parts = candidates[0]["content"]["parts"]


    print(json.dumps(parts, indent=2))

    for part in parts:

        if "text" in part:
            
            reply_json = json.loads(part["text"])
            move = reply_json["move"]
            reply = reply_json["message"]


            print(reply)

            send_chat_message(reply, sender=AGENT_NAME, move=move )




    print("Waiting 15 seconds...")
    time.sleep(15)

print("\n⏱ 2 minutes elapsed. Stopping agent.")