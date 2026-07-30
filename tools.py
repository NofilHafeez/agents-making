import json

CHAT_FILE = "chats.json"


def get_chat_messages():
    print("get_chat_messages() called")
    """
    Read all chat messages.
    """

    with open(CHAT_FILE, "r") as f:
        return {
            "messages": json.load(f)
        }


def send_chat_message(message: str, sender: str, move: str):
    print("send_chat_message() called")
    print("Message:", message)
    """
    Append a new message.
    """

    with open(CHAT_FILE, "r") as f:
        chats = json.load(f)

    chats.append(
        {
            "sender": sender,
            "move":   move,
            "message": message
        }
    )

    with open(CHAT_FILE, "w") as f:
        json.dump(chats, f, indent=4)

    return {
        "status": "success",
        "move":   move,
        "message": message
    }


def findProbabofMoves(messages: list, my_name: str):

    stats = is_cooperative(messages, my_name)

    total = stats["coop"] + stats["hostile"]

    if total == 0:
        return {
            "probabDef": 0.0,
            "probabCoop": 0.0
        }

    probabDef = stats["hostile"] / total
    probabCoop = stats["coop"] / total

    return {
        "probabDef": probabDef,
        "probabCoop": probabCoop
    }


def is_cooperative(messages: list, my_name: str):

    hostile = 0
    coop = 0

    for msg in messages:

        if msg["sender"] == my_name:
            continue

        text = msg["message"].lower()

        cooperate_count = sum(
            1 for m in messages
            if m["sender"] != my_name and m["move"] == "COOPERATE"
        )

        defect_count = sum(
            1 for m in messages
            if m["sender"] != my_name and m["move"] == "DEFECT"
        )

    return {
        "coop": cooperate_count,
        "hostile": defect_count
    }

def get_strategy(messages, my_name):

    # Find last opponent message
    for msg in reversed(messages):

        if msg["sender"] != my_name:

            text = msg["message"].lower()

            hostile_words = [
                "hate","idiot","stupid",
                "fight","attack","shut",
                "leave","ignore"
            ]

            hostile = any(word in text for word in hostile_words)

            if hostile:
                return "Reply in the same hostile tone."

            return "Reply cooperatively."

    return "Start cooperatively."