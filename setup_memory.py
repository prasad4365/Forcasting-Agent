import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

load_dotenv()

print("1. Keys Load කරගන්නවා...")
vector_store_address: str = os.getenv("AZURE_SEARCH_ENDPOINT")
vector_store_password: str = os.getenv("AZURE_SEARCH_ADMIN_KEY")
index_name: str = "fuel-news-index"

# Embedding Model එක සෙට් කරනවා
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small", # මේ නම අනිවාර්යයෙන්ම AI Studio එකේ තියෙන්න ඕනේ
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

print("\n2. Azure OpenAI Embedding එක Test කරනවා...")
try:
    test_result = embeddings.embed_query("Hello World")
    print(" -> ✅ OpenAI Embeddings වැඩ කරනවා! (කිසිම අවුලක් නෑ)")
except Exception as e:
    print(f" -> ❌ OPENAI ERROR එකක්: {e}")
    print("💡 විසඳුම: ඔයාගේ Azure AI Studio එකේ 'Deployments' වලට ගිහින් බලන්න 'embedding-model' කියලා නමක් තියෙනවද කියලා. නැත්නම් අලුතෙන් text-embedding-3-small එකක් ඒ නමින් deploy කරන්න.")
    exit() # මෙතනින් කෝඩ් එක නවත්තනවා


print("\n3. Azure AI Search එකට කනෙක්ට් වෙනවා...")
try:
    vector_store = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
    )
    print(f" -> ✅ Success! Index '{index_name}' හැදුවා. ඔක්කොම ගොඩ!")
except Exception as e:
    print(f" -> ❌ AZURE SEARCH ERROR එකක්: {e}")
    print("💡 විසඳුම: ඔයාගේ .env ෆයිල් එකේ AZURE_SEARCH_ENDPOINT එක හරියටම https://ඔයාගේ-නම.search.windows.net වගේ තියෙනවද බලන්න. අගට / කෑලි එන්න බෑ.")