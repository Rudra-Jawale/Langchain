from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
    
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_history = [
    SystemMessage(content="You are a helpful assistant."),
]

def chat_with_gemini(user_input):
    chat_history.append(HumanMessage(content = user_input))
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content = response.content.strip()))
    return response.content.strip()

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        # chat_history.append(user_input)
        if user_input.lower() in ["quit", "exit", "bye"]:
            break

        response = chat_with_gemini(user_input)
        # chat_history.append(response)
        print("Chatbot:", response)

    print(chat_history)