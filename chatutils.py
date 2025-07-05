
from langchain.schema import HumanMessage, AIMessage


# --------------- Intent Rules ---------------

GREETING_KEYWORDS = ["hi", "hello", "hey", "good morning", "good evening", "thanks", "thank you", "appreciate"]
PRICE_KEYWORDS = ["how much", "price", "cost", "fees", "charges"]

def is_greeting_or_compliment(text):
    return any(word in text.lower() for word in GREETING_KEYWORDS) and len(text.split()) <= 6

def is_price_question(text):
    return any(word in text.lower() for word in PRICE_KEYWORDS)

def random_greeting_response():
    import random
    return random.choice([
        "Hello! this lara,Ai engineer  😊 How can I help you today?",
        "Hi there! this lara,Ai engineer,👋",
        "Hey! this lara,Ai engineer,Need anything about AI",
        "Welcome! this lara,Ai engineer, What would you like to ask about AI?"
    ])

def compress_chat_history(chat_history, max_turns=5, max_length=300):
    def is_small_talk(msg):
        small_talk_phrases = [
            "hi", "hello", "thanks", "thank you", "how are you", "good morning",
            "bye", "okay", "cool", "yes", "no", "hmm"
        ]
        return any(phrase in msg.content.lower() for phrase in small_talk_phrases)

    # Only keep the last `max_turns * 2` messages
    trimmed = chat_history[-max_turns * 2:]

    compressed = []
    for msg in trimmed:
        if is_small_talk(msg):
            continue  # Skip small talk

        # Trim long messages
        content = msg.content[-max_length:] if len(msg.content) > max_length else msg.content

        if isinstance(msg, HumanMessage):
            compressed.append(HumanMessage(content=content))
        elif isinstance(msg, AIMessage):
            compressed.append(AIMessage(content=content))
        else:
            compressed.append(msg)

    return compressed