# from langchain.agents import AgentType,initialize_agent,load_tools
# from langchain.prompts import ChatPromptTemplate
# from langchain.output_parsers import ResponseSchema,StructuredOutputParser
# # from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOllama

# import os
# os.environ["SERPAPI_API_KEY"]="f5a17d267ce285ea7b9a02e871574ce34a0980c95a8f7bd218f3bcad776f05f3"
# tools=load_tools(["serpapi"])
# llm=ChatOllama(model="mistral",temperature=0.0)
# brand_name=ResponseSchema(name="brand_name",description="this is the brand of the product")
# product_name=ResponseSchema(name="product_name",description="this is the product name")
# description=ResponseSchema(name="description",description="this is the short description of the product")
# product_price=ResponseSchema(name="price",description="this will be in number, represents the price of the product")
# product_rating=ResponseSchema(name="rating",description="this is whole integer,this gives the rating between 1-10")
# response_schema=[brand_name,product_name,description,product_price,product_rating]
# output_parser=StructuredOutputParser.from_response_schemas(response_schema)
# format_instruction=output_parser.get_format_instructions()
# ts="""
# You are an intelligent search master and analyst who can search internet using serpapi tool and analyse any product to find the brand of the product ,name of the product,
# product description,price and rating between 1-5 based on your owen analysis.
# Take the input below delimited by tripe backticks and use it to search and analyse using serapi tool
# input:```{input}```
# {format_instruction}
# """
# prompt=ChatPromptTemplate.from_template(ts)
# fs=prompt.format_messages(input="best android phone in India",format_instruction=format_instruction)
# agent=initialize_agent(tools,llm,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
# response=agent.run(fs)
# output=output_parser.parse(response)
# print(output)
# # print(output["brand_name"],output["product_name"])
from langchain.agents import AgentType,initialize_agent,load_tools
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema,StructuredOutputParser
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

import os
os.environ["SERPAPI_API_KEY"]="f5a17d267ce285ea7b9a02e871574ce34a0980c95a8f7bd218f3bcad776f05f3"
tools=load_tools(["serpapi"])
llm=ChatOllama(model="mistral",temperature=0.0)

university_name=ResponseSchema(name="university_name",description="this represents the university name or institution name or organization name or hospital name")

contact_number=ResponseSchema(name="contact_number",description="this represents the contact number of the university or represents the contact number of institution or represents the contact number of hospital")

email_address=ResponseSchema(name="email_address",description="this represents the email address of the university or represents the email address of institution or represents the email address of hospital")

physical_address=ResponseSchema(name="physical_address",description="this represents the physical address of the university or represents the physical address of institution or represents the physical address of hospital")
#we can add whatever the entiti has to be collected by the llm here

# product_rating=ResponseSchema(name="rating",description="this is whole integer,this gives the rating between 1-10")
response_schema=[university_name,contact_number,email_address,physical_address]
output_parser=StructuredOutputParser.from_response_schemas(response_schema)
format_instruction=output_parser.get_format_instructions()
ts="""
You are an intelligent search master and analyst who can search internet using serpapi tool and analyse any website to find university name,contact number,email address,physical address based on your owen analysis.
Take the input below delimited by tripe backticks and use it to search and analyse using serapi tool
input:```{input}```
{format_instruction}
"""
prompt=ChatPromptTemplate.from_template(ts)
fs=prompt.format_messages(input="only provide me the following details like university name,contact number,email address,physical address of aiims",format_instruction=format_instruction)
agent=initialize_agent(tools,llm,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
response=agent.run(fs)
output=output_parser.parse(response)
print(output)
# print(output["brand_name"],output["product_name"])