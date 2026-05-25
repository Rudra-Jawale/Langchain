from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
    
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def chat_with_gpt(prompt):
    response = model.invoke(prompt)
    return response.content.strip()

if __name__ == "__main__":
    while True:
        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit", "bye"]:
            break

        response = chat_with_gpt(user_input)
        print("Chatbot:", response)