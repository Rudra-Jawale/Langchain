from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(gt = 18 ,description="The age of the person")
    city: str = Field(description="The city of the person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = 'Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables = ['topic'],
    partial_variables = {'format_instruction':parser.get_format_instructions()} )

chain = template | model | parser
result = chain.invoke({"topic": ""})
print(result)