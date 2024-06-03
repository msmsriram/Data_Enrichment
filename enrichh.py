import streamlit as st
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
from langchain_community.vectorstores import FAISS
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache


with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Data Enricher")
    st.info("Welcome to the Data Enricher! This tool is designed to provide enriched data about various affiliations.By simply entering the name of an affiliation, the application presents a wealth of information")

input_data=st.text_input(label="Enter an Affiliation") 
if input_data!="":

    # model_local = ChatOllama(model="mistral",temperature=0)
    # set_llm_cache(InMemoryCache())
    
    def sample(name):
        # df = pd.DataFrame(columns=["university_name","address", "email_address", "contact_number"])
        
        if name=='LUMC':
            model_local = ChatOllama(model="mistral",temperature=0)
            set_llm_cache(InMemoryCache())
            embedding = embeddings.OllamaEmbeddings(model='nomic-embed-text')

            vectorstore = FAISS.load_local("db_demos/test_9", embedding,allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever()
            query="provide the metioned details from the context and make sure you providing them from the given context"
            new_docs = vectorstore.similarity_search(query)
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
            

            json_output = after_rag_chain.invoke("provide the metioned details from the context and make sure you providing them from the given context")
            # print(json_output)
            # dff = pd.json_normalize(json_output)
            # print(dff)
            # df = pd.concat([df, dff], ignore_index=True)
            # df
        elif name=='Makerere University College Of Health Sciences':
            embedding = embeddings.OllamaEmbeddings(model='nomic-embed-text')

            vectorstore = FAISS.load_local("db_demos/test_11", embedding,allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever()
            query="provide the metioned details from the context and make sure you providing them from the given context"
            new_docs = vectorstore.similarity_search(query)
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
            set_llm_cache(InMemoryCache())

            json_output = after_rag_chain.invoke("provide the metioned details from the context and make sure you providing them from the given context")
            # print(json_output)
            # dff = pd.json_normalize(json_output)
            # print(dff)
            # df = pd.concat([df, dff], ignore_index=True)
            # df
        elif name=='University of Zimbabwe':
            embedding = embeddings.OllamaEmbeddings(model='nomic-embed-text')

            vectorstore = FAISS.load_local("db_demos/test_13", embedding,allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever()
            query="provide the metioned details from the context and make sure you providing them from the given context"
            new_docs = vectorstore.similarity_search(query)
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
            json_output = after_rag_chain.invoke("provide the metioned details from the context and make sure you providing them from the given context")
            
            # print(json_output)
            # dff = pd.json_normalize(json_output)
            # print(dff)
            # df = pd.concat([df, dff], ignore_index=True)
            # df
        elif name=='University of Western Australia':
            embedding = embeddings.OllamaEmbeddings(model='nomic-embed-text')

            vectorstore = FAISS.load_local("db_demos/test_6", embedding,allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever()
            query="provide the metioned details from the context and make sure you providing them from the given context"
            new_docs = vectorstore.similarity_search(query)
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
            json_output = after_rag_chain.invoke("provide the metioned details from the context and make sure you providing them from the given context")
            # print(json_output)
            # dff = pd.json_normalize(json_output)
            # print(dff)
            # df = pd.concat([df, dff], ignore_index=True)
            # df
        elif name=='School of Public Health, Nanjing Medical University':
            embedding = embeddings.OllamaEmbeddings(model='nomic-embed-text')

            vectorstore = FAISS.load_local("db_demos/test_4", embedding,allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever()
            query="provide the metioned details from the context and make sure you providing them from the given context"
            new_docs = vectorstore.similarity_search(query)
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
            json_output = after_rag_chain.invoke("provide the metioned details from the context and make sure you providing them from the given context")
            # print(json_output)
            # dff = pd.json_normalize(json_output)
            # print(dff)
            # df = pd.concat([df, dff], ignore_index=True)
            # df
        elif name=='aiims':
            embedding = embeddings.OllamaEmbeddings(model='nomic-embed-text')

            vectorstore = FAISS.load_local("db_demos/aiims", embedding,allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever()
            query="provide the metioned details from the context and make sure you providing them from the given context"
            new_docs = vectorstore.similarity_search(query)
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
            json_output = after_rag_chain.invoke("provide the metioned details from the context and make sure you providing them from the given context")
            # print(json_output)
            # dff = pd.json_normalize(json_output)
        return json_output

#-------------------------------Output-----------------------------------
    st.title("Enriched Data:-")
    with st.spinner('Processing...'):
        output_data = sample(input_data)
    
    st.markdown(f"**University Name** : {output_data['university_name']}")
    st.markdown(f"**Address** : {output_data['address']}")
    st.markdown(f"**Email-ID** : {output_data['email_address']}")
    st.markdown(f"**Contact No** : {output_data['contact_number']}")
    # st.write(f"relevant source : {source}")
    # st.write(output_data)
