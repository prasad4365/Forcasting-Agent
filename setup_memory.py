import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

load_dotenv()

print("1. Keys Loading...")
vector_store_address: str = os.getenv("AZURE_SEARCH_ENDPOINT")
vector_store_password: str = os.getenv("AZURE_SEARCH_ADMIN_KEY")
index_name: str = "fuel-news-index"

# Embedding Model 
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small", 
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

print("\n2. Azure OpenAI Embedding tesing")


print("\n3. Azure AI Search ")

vector_store = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
)
print(f" -> Success! Index '{index_name}'")