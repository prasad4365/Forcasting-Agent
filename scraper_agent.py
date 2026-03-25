import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool 
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

load_dotenv()

# --- 1. මතකය (Database) සෙට් කිරීම ---
print("Connecting to Memory...")
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small", # <--- මෙතන ඔයාගේ හරි නම දාන්න 
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_ADMIN_KEY"),
    index_name="fuel-news-index",
    embedding_function=embeddings.embed_query,
)

# --- 2. Tools හැදීම (වඩාත් ආරක්ෂිත Class-based ක්‍රමයට) ---
duckduckgo = DuckDuckGoSearchRun()

class SearchInternetTool(BaseTool):
    name: str = "Search_Internet"
    description: str = "Search the internet for news, oil prices, and information."

    def _run(self, query: str) -> str:
        return duckduckgo.invoke(query)

class SaveToMemoryTool(BaseTool):
    name: str = "Save_To_Memory"
    description: str = "Useful to save highly important global crude oil news into the database."

    def _run(self, news_text: str) -> str:
        vector_store.add_texts([news_text])
        return "Successfully saved the news to the Azure Vector Database!"

search_tool = SearchInternetTool()
save_to_memory = SaveToMemoryTool()

# --- 3. Agent ගේ මොළය සෙට් කිරීම ---
print("Waking up the Agent Brain...")
azure_llm = LLM(
    model="azure/gpt-5.4-mini", 
    api_key=os.getenv("AZURE_OPENAI_SEARCH_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_SEARCH_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_SEARCH_API_VERSION")
)

# --- 4. Agent ව හැදීම ---
scraper_agent = Agent(
    role='Global Energy News Analyst',
    goal='Search the web for the latest global crude oil prices and news, and save the most important updates to memory.',
    backstory="You are an expert energy market analyst. You find reliable news about oil prices, geopolitical events affecting fuel, and save summaries for future forecasting.",
    tools=[search_tool, save_to_memory], 
    llm=azure_llm,
    verbose=True,
    allow_delegation=False
)

# --- 5. Task එක දීම ---
task1 = Task(
    description="Search the web for today's top 2 news articles about global crude oil prices or geopolitical tensions affecting oil. Summarize them and STRICTLY use the Save_To_Memory tool to store the summary.",
    expected_output="A confirmation message that the news has been found and saved to memory.",
    agent=scraper_agent
)

# --- 6. වැඩේ පටන් ගැනීම ---
crew = Crew(
    agents=[scraper_agent],
    tasks=[task1],
    process=Process.sequential
)

print("\n🚀 Agent is starting the research...\n")
result = crew.kickoff()

print("\n=== Done ===")
print(result)