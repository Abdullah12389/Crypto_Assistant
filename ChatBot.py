from langchain_community.utilities import SQLDatabase
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain import hub
import os
load_dotenv()
mistralapi=os.environ['MISTRAL']
taviliy=os.environ['TAVILY']

llm=init_chat_model("ministral-8b-latest",model_provider="mistralai",api_key=mistralapi)

db=SQLDatabase.from_uri("sqlite:///db/bitcoin_data.db")
toolkit=SQLDatabaseToolkit(db=db,llm=llm)
sqltools=toolkit.get_tools()
table_info=db.get_table_info()
sqlprompt=hub.pull("langchain-ai/sql-agent-system-prompt")
sqlprompt=sqlprompt.format(dialect="SQLite", top_k=5)

class GraphState(TypedDict):
    userinput:str
    movewhere:str
    aimessages:list[str]
    query:str
    finalanswer:str


def router(state:GraphState):
    userinput=state["userinput"]
    prompt=ChatPromptTemplate.from_messages([
        ("system","You are assistant who have to decide what user need right now reponsd with 'News' or 'Analysis' or 'Research' or 'all' 'none' dont use any puctuation, symbols or extra text in the reponse. Remember when you respond with analysis we run sql queries on a database with bitcoin high low volume and rsi mcad etc respond this only when the user asks about statistical analysis on the bitcoin data. When you select Research we search internet for any financial oredictions or articles. You must first understand prompt and check which option is best for better responses. Respond with 'all' when i the query of user is diverse like getting some financial advice or require some extra info other than seraching old database. If none of the options is required or question is not about bitcoin or related terms or personalities respond 'none'"),
        ("user",f"Here is the input from user '{userinput}'")
    ])
    chain=prompt | llm
    response=chain.invoke({'userinput':userinput})
    print(response.content)
    state['movewhere']=response.content.lower()
    return state

def querygen(state:GraphState):
    userinput=state['userinput']
    prompt=ChatPromptTemplate.from_messages([
        ("system","You are a assistant who would generate questions for an agent which generates sql queries for data {datainfo}. Analyze the user input and generate the questions for that agent that gives maximum information that user need for better answer. You should ask only 5 statistical questions without heading just simple questions"),
        ("user","Here is a prompt from user '{userinput}'")
    ])
    chain=prompt | llm
    response=chain.invoke({"datainfo":table_info,'userinput':userinput})
    state['query']=response.content
    return state

def news(state:GraphState):
    news_update=TavilySearch(
        tavily_api_key=taviliy,
        max_results=5,
        topic="news"
    )   
    userinput=state['userinput']
    searchagent=create_react_agent(llm,tools=[news_update])
    response=searchagent.invoke({"messages":[{"role":"user","content":userinput}]})
    state['aimessages'].append(response['messages'][-1].content)
    return state

def analysis(state:GraphState):
    queries=state['query']
    sqlagent=create_react_agent(llm,tools=sqltools,prompt=sqlprompt)
    response=sqlagent.invoke({"messages":[{"role":"user","content":queries}]})
    state['aimessages'].append(response['messages'][-1].content)
    return state

def search(state:GraphState):
    userinput=state['userinput']
    finance_update=TavilySearch(
        tavily_api_key=taviliy,
        max_results=5,
        topic="finance"
    )
    searchagent=create_react_agent(llm,tools=[finance_update])
    response=searchagent.invoke({"messages":[{"role":"user","content":userinput}]})
    state['aimessages'].append(response['messages'][-1].content)
    return state

def finalnode(state:GraphState):
    userinput=state['userinput']
    aimessages=state['aimessages']
    prompt=ChatPromptTemplate.from_messages([
        ("system","You are an agent who have to write final comprehensive answer based on user query and the provided docs given by different AI agents here are some docs that might help {aimessages}"),
        ("user","{userinput}")
    ])
    chain=prompt | llm
    response=chain.invoke({"aimessages":aimessages,"userinput":userinput})
    state['finalanswer']=response.content
    return state

def runall(state:GraphState):
    return state

def random(state:GraphState):
    response=llm.invoke(state['userinput'])
    state['finalanswer']=response.content
    return state

builder=StateGraph(GraphState)
builder.add_node("decide",router)
builder.add_node("random",random)

builder.add_edge(START,"decide")
builder.add_conditional_edges("decide",lambda state:state['movewhere'],
{
  "news":"news",
  "analysis":"querygen",
  "research":"search",
  "none":"random",
  "all":"runall"
})
builder.add_sequence([
    ("runall",runall),
    ("queerygenall",querygen),
    ("analyzeall",analysis),
    ("newsall",news),
    ("searchall",search),
    ("finalnodeall",finalnode)
])
builder.add_sequence([("search",search),("finalsearch",finalnode)])
builder.add_sequence([("news",news),("finalnews",finalnode)])
builder.add_sequence([("querygen",querygen),("analyze",analysis),("finalanalyze",finalnode)])
builder.add_edge("random",END)

graph=builder.compile()
