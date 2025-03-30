import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from IPython.display import Image, display

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Initialize Tavily search tool
search_tool = TavilySearch(
  tavily_api_key=os.getenv("TAVILY_API_KEY"),
  max_results=5,
  topic="news",
)

# Define state type
class State(TypedDict):
  messages: Annotated[list, add_messages]

# Initialize state graph
graph_builder = StateGraph(State)

# Initialize LLM
llm = ChatOpenAI(
  model="deepseek/deepseek-r1",
  temperature=0,
  openai_api_key=os.getenv("OPENROUTER_API_KEY"),
  base_url="https://openrouter.ai/api/v1",
)

# Define LLM Node
class LLMNode:
  def __init__(self, llm: ChatOpenAI):
    self.llm = llm

  def __call__(self, state: State):
    return {'messages': [self.llm.invoke(state['messages'])]}

# Initialize nodes
llm_node = LLMNode(llm.bind_tools([search_tool]))
tool_node = ToolNode([search_tool])

# Build the graph
graph_builder.add_node('llm', llm_node)
graph_builder.add_node('tools', tool_node)

graph_builder.add_edge(START, 'llm')
graph_builder.add_conditional_edges('llm', tools_condition)
graph_builder.add_edge('tools', 'llm')

# Compile the agent
agent = graph_builder.compile()

def chat_with_agent(query: str):
  """
  Send a query to the agent and print the conversation steps
  """
  for step in agent.stream(
    {"messages": query},
    stream_mode="values"
  ):
    step['messages'][-1].pretty_print()

if __name__ == "__main__":
  # Example usage
  while True:
    try:
      user_input = input("\nEnter your question (or 'quit' to exit): ")
      if user_input.lower() in ['quit', 'exit', 'q']:
        break
      chat_with_agent(user_input)
    except KeyboardInterrupt:
      print("\nExiting...")
      break 