from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


model_local = ChatOllama(model="mistral")

# 1. Split data into chunks
# urls = [
#     "https://ollama.com/",
#     "https://ollama.com/blog/windows-preview",
#     "https://ollama.com/blog/openai-compatibility",
# ]
# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]
# loader = BSHTMLLoader("C:\\Users\\Barani\\Desktop\\local_ollama\\llm_testing_urls\\test_4.html")
# loader = BSHTMLLoader("C:\\Users\\Barani\\Desktop\\local_ollama\\mayo.html")
# loader = BSHTMLLoader("C:\\Users\\Barani\\Desktop\\local_ollama\\llm_testing_urls\\test_3.html")

# data = loader.load()
# print(data)

loader = BSHTMLLoader("C:\\Users\\Barani\\Desktop\\local_ollama\\llm_testing_urls\\test_8.html", 
                       bs_kwargs={"features": "html.parser"})
data=loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=650, chunk_overlap=100)
doc_splits = text_splitter.split_documents(data)
print(len(doc_splits))
# 2. Convert documents to Embeddings and store them
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
)
retriever = vectorstore.as_retriever()

# # 3. Before RAG
# print("Before RAG\n")
# before_rag_template = "Provide me {topic}"
# before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
# before_rag_chain = before_rag_prompt | model_local | StrOutputParser()
# print(before_rag_chain.invoke({"topic": "the addresses of mayo clinic which is in Rochester"}))


class format_json(BaseModel):
    university_name : str = Field(description="organization name from the given context")
    address: str = Field(description="address from the given context")
    email_address: str = Field(description="email address from the given context")
    contact_number: str = Field(description="contact number from the given context")
    


# 4. After RAG
print("\n########\nAfter RAG\n")
after_rag_template = """Answer the question based only on the following context:
{context} and only provide these details in this order only and the response will always will be containing these informations only such as address,contact number,email address in a json format
{format_instructions}
Question: {question}
"""
parser = JsonOutputParser(pydantic_object=format_json)
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template,partial_variables={"format_instructions": parser.get_format_instructions()},)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | parser
)

# after_rag_chain = (
# {"context": retriever, "question": RunnablePassthrough()}
# | after_rag_prompt
# | model_local
# | JsonOutputParser(
#         template="""
#         {
#             "address": "{address}",
#             "contact number": "{contact_number}",
#             "email address": "{email_address}",
#             "zip code": "{zip_code}"
# }"""
# )
# )
print(after_rag_chain.invoke("provide the metioned details from the context and make sure you providing them from the given context"))