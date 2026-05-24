from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
model_id = "meta-llama/Llama-3.1-8B-Instruct",
task = "conversational",
pipeline_kwargs = dict(
    temperature = 0.5,
    max_new_tokens = 100
)
)

model = ChatHuggingFace(llm = llm)
result = model.invoke("Who is the prime minister of India?")
print(result.content)