import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("GITHUB_TOKEN_OPENAI"), base_url=os.getenv("endpoint"))

class ChatSession:
    def __init__(self):
        self.history = [
            {
                "role": "system",
                "content": "You are a pharmacy assistant. Provide accurate and concise information about medications, dosages, side effects, and interactions."
            }
        ]

    def chat(self, user_input):
        self.history.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=self.history,
            stream=True,
            temperature=1.0,
            top_p=1.0,
            max_tokens=1000,
        )

        assistant_reply = ""

        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    assistant_reply += delta.content
                    print(delta.content, end='', flush=True)

        self.history.append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    chat_session = ChatSession()
    
    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        print("Assistant: ", end='', flush=True)
        chat_session.chat(user_input)





