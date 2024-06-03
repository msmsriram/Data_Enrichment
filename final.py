import os
import pandas as pd
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

# Initialize DataFrame
df = pd.DataFrame(columns=["university_name","address", "email_address", "contact_number"])
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
# Directory containing HTML files
directory = "C:\\Users\\Barani\\Desktop\\local_ollama\\test"

# Iterate over each HTML file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".html"):
        filepath = os.path.join(directory, filename)
        model_local = ChatOllama(model="gemma")

        # Load and split data
        loader = BSHTMLLoader(filepath, bs_kwargs={"features": "html.parser"})
        data = loader.load()
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=250)


        doc_splits = text_splitter.split_documents(data)
        print("length of doc : ",len(doc_splits))

        # Convert documents to Embeddings and store them
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
        )
        retriever = vectorstore.as_retriever()
        class format_json(BaseModel):
            university_name : str = Field(description="organization name from the given context")
            address: str = Field(description="address from the given context")
            email_address: str = Field(description="email address from the given context")
            contact_number: str = Field(description="contact number from the given context")

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
        # Generate JSON output
        json_output = after_rag_chain.invoke("provide the metioned details and make sure you providing them from the given context")
        print(json_output)
        # like university_name,address,email_address,contact_number from the context
        # Convert JSON output to DataFrame row and append to DataFrame
        # df = df.append(json_output, ignore_index=True)
        dff = pd.json_normalize(json_output)
        df = pd.concat([df, dff], ignore_index=True)


# Show DataFrame
print(df)
