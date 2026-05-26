from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema




load_dotenv()



model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


schema = [
    ResponseSchema(name="fact_1", description="The name of the person"),
    ResponseSchema(name="fact_2", description="The age of the person"),
    ResponseSchema(name="fact_3", description="The city of the person")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables = ['topic'],
    partial_variables = {'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({"topic": ""})
chain = template | model | parser
result = chain.invoke({"topic": ""})
print(result)