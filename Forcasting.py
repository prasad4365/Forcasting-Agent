from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import smtplib
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool 
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

load_dotenv()

print("Connecting to Memory...")
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small", # <--- Embedding නම
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


class ReadFromMemoryTool(BaseTool):
    name: str = "Read_From_Memory"
    description: str = "Search the database for recent global crude oil news to base forecasts on."

    def _run(self, query: str) -> str:
        docs = vector_store.similarity_search(query, k=3) 
        if not docs:
            return "No relevant news found in memory."
        return "\n\n".join([doc.page_content for doc in docs])

search_tool = SearchInternetTool()
save_to_memory = SaveToMemoryTool()
read_from_memory = ReadFromMemoryTool()

print("Waking up the Agent Brain...")
azure_llm = LLM(
    model="azure/gpt-5.4-mini",
    api_key=os.getenv("AZURE_OPENAI_SEARCH_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_SEARCH_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_SEARCH_API_VERSION")
)



scraper_agent = Agent(
    role='Global Energy News Analyst',
    goal='Search the web for the latest global crude oil prices and news, and save the most important updates to memory.',
    backstory="You find reliable news about oil prices and save summaries to the database.",
    tools=[search_tool, save_to_memory], 
    llm=azure_llm,
    verbose=True,
    allow_delegation=False
)


forecast_agent = Agent(
    role='Senior Fuel Market Forecaster',
    goal='Analyze recent crude oil news from memory and predict short-term price movements.',
    backstory="You are a veteran oil market analyst. You read stored news and provide accurate, realistic price forecasts based on geopolitical and economic factors.",
    tools=[read_from_memory], 
    llm=azure_llm,
    verbose=True,
    allow_delegation=False
)



task1 = Task(
    description="Search the web for today's top news about global crude oil prices. Summarize them and STRICTLY use the Save_To_Memory tool to store the summary.",
    expected_output="A confirmation message that the news has been found and saved to memory.",
    agent=scraper_agent
)


task2 = Task(
    description="Use the Read_From_Memory tool to search for 'crude oil prices and geopolitical tensions'. Based on the retrieved information, write a short, professional forecast report predicting if fuel prices will rise, fall, or stabilize in the next few weeks and what is the tomorrow crude oil price in USD.",
    expected_output="A 2-3 paragraph forecast report detailing the expected price trend and the reasons behind it with a specific price prediction for tomorrow.",
    agent=forecast_agent
)


crew = Crew(
    agents=[scraper_agent, forecast_agent], 
    tasks=[task1, task2], 
    process=Process.sequential 
)

print("\n System is starting the Multi-Agent workflow...\n")
result = crew.kickoff()

print("\n==============================================")
print("FINAL FORECAST REPORT):")
print("==============================================\n")
print(result)



def send_forecast_email(forecast_text):
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    receiver_email = os.getenv("RECEIVER_EMAIL")

    if not sender_email or not sender_password or not receiver_email:
        print("\n .env ෆයිල් එකේ Email විස්තර නැති නිසා Email එක යැව්වේ නෑ.")
        return

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "AI Agent: Global Fuel Price Forecast Report"
 
    msg.attach(MIMEText(str(forecast_text), 'plain'))

    try:
        print(f"\n Sending email to {receiver_email}...")
       
        server = smtplib.SMTP('smtp.gmail.com', 587) 
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error: {e}")

send_forecast_email(result)