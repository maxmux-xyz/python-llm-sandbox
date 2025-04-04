import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import json
import logging

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Model configuration
MODEL = "deepseek/deepseek-chat-v3-0324"

# Initialize LLM
llm = ChatOpenAI(
    model=MODEL,
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Type definitions
class State(TypedDict):
    text: str  # Stores the original input text
    classification: str  # Represents the classification result (e.g., category label)
    human_entities: List[str]  # Holds a list of extracted human entities
    company_entities: List[str]  # Holds a list of extracted company entities
    summary: str  # Stores a summarized version of the text

def classification_node(state: State):
    """
    Classify the text into one of predefined categories.
    
    Parameters:
        state (State): The current state dictionary containing the text to classify
        
    Returns:
        dict: A dictionary with the "classification_type" key containing the category result
        
    Categories:
        - News: Host(s) report on current events. Factual reporting. Traditional news media.
        - Finance: Host(s) discuss about finance, markets, economics, etc.
        - Education: Host(s) discuss about education, learning, teaching, etc.
        - Podcast: Discussion between hosts, about one or more subjects.
        - Other: Content that doesn't fit the above categories
    """
    logger.info("Starting classification...")
    
    instructions = """
    You are a helpful assistant that can classify youtube transcripts into one of the following categories:
    - News: Factual reporting of current events
    - Podcast: Discussion between hosts, about one or more subjects
    - Finance: Discussion about finance, markets, economics, etc. Could be one or multiple hosts.
    - Education: Discuss about education, learning, teaching, etc. Could be one or multiple hosts.
    - Other: Content that doesn't fit the above categories

    return 'news', 'podcast', 'finance', 'education', or 'other' - no other words or characters.
    Output should be in the form of a single string, eg: 'news'
    """

    prompt = PromptTemplate(
        input_variables=["text"],
        template=instructions + "\n\nText: {text}\n\nCategory:"
    )

    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    logger.info(f"Classification result: {classification}")
    return {"classification": classification}

# def human_entity_extraction_node(state: State):
#     """Extract human entities from the text."""
#     logger.info("Starting human entity extraction...")
    
#     instructions = """
#     Extract all the Human entities from the following transcript. 
#     Provide the result as a comma-separated list.

#     The output should be a comma-separated list of human entities, no other text or characters.
#     Parsable by a python script.
#     """
#     prompt = PromptTemplate(
#         input_variables=["text"],
#         template=instructions + "\n\nText: {text}\n\nHuman Entities:"
#     )
    
#     message = HumanMessage(content=prompt.format(text=state["text"]))
#     human_entities = llm.invoke([message]).content.strip().split(", ")
#     logger.info(f"Extracted {len(human_entities)} human entities")
#     return {"human_entities": human_entities}

def company_entity_extraction_node(state: State):
    """Extract company entities from the text."""
    logger.info("Starting company entity extraction...")
    
    instructions = """
    Extract all the Company entities from the following transcript. 
    Provide the result as a comma-separated list.

    The output should be a comma-separated list of company entities, no other text or characters.
    Parsable by a python script.
    """
    prompt = PromptTemplate(
        input_variables=["text"],
        template=instructions + "\n\nText: {text}"
    )
    
    message = HumanMessage(content=prompt.format(text=state["text"]))
    company_entities = llm.invoke([message]).content.strip().split(", ")
    logger.info(f"Extracted {len(company_entities)} company entities")
    return {"company_entities": company_entities}

def summarize_finance_node(state: State):
    """Summarize finance-related content."""
    logger.info("Starting finance content summarization...")
    summarization_prompt = PromptTemplate.from_template(
        """You are a helpful assistant that summarizes transcripts of youtube videos. 
        
        Summarize the transcript in a way that is easy to understand and contains the most important information. 
        - First section should be a high level summary - aim for 3-5 concise bullet points.
        - Second section should be to list the speakers and the main points they make.
        - Third section should be a list of topics that are discussed.
        - Fourth section should be to flag if any stock tickers or cryptocurrencies are mentioned. If so, what was the sentiment on them?
        - Fifth section should be about markout outlook. Are speakers bullish or bearish - short or long term? Only if transcript is about markets or trading.
        - Sixth section should be to flag any other interesting information that is not covered in the other sections, and / or perhpas one or a few meanigful quotes from any of the speakers.
        
        Keep it short and concise, each section should be well apparent and contain up to 10 bullet points.
        
        Text: {text}
        
        Finance Summary:"""
    )
    
    chain = summarization_prompt | llm
    response = chain.invoke({"text": state["text"]})
    return {"summary": response.content}

def summarize_news_node(state: State):
    """Summarize news content."""
    summarization_prompt = PromptTemplate.from_template(
        """You are a helpful assistant that summarizes transcripts of youtube videos.

        You are summarizing a news transcript.
        
        Summarize in a way that is easy to understand and contains the most important information. 
        - First section should be a high level summary - aim for 3-5 concise bullet points.
        - Second section should be a list of topics that are discussed.
        - Third section should be to flag any other interesting information that is not covered in the other sections, and / or perhpas one or a few meanigful quotes from any of the speakers.

        Keep it short and concise, each section should be well apparent and contain up to 10 bullet points.
        
        Text: {text}
        
        News Summary:"""
    )
    
    chain = summarization_prompt | llm
    response = chain.invoke({"text": state["text"]})
    return {"summary": response.content}

def summarize_podcast_node(state: State):
    """Summarize podcast content."""
    summarization_prompt = PromptTemplate.from_template(
        """You are a helpful assistant that summarizes transcripts of youtube videos.

        You are summarizing a podcast transcript. Two or more hosts are discussing one or more subjects.
        
        Summarize in a way that is easy to understand and contains the most important information. 
        - First section should be a list of topics that are discussed.
        - Second section should be a small summary of each topics discussed.
        - Third section should be a list of speakers and their main points. In order of who contributed to most to the less.
        - Fourth section should be to flag any other interesting information that is not covered in the other sections, and / or perhpas one or a few meanigful quotes from any of the speakers.

        Keep it short and concise, each section should be well apparent and contain up to 10 bullet points.
        
        Text: {text}
        
        Podcast Summary:"""
    )
    
    chain = summarization_prompt | llm
    response = chain.invoke({"text": state["text"]})
    return {"summary": response.content}

def summarize_education_node(state: State):
    """Summarize educational content."""
    summarization_prompt = PromptTemplate.from_template(
        """You are a helpful assistant that summarizes transcripts of youtube videos.

        You are summarizing an educational video transcript.
        
        Summarize in a way that is easy to understand and contains the most important information. 
        - First section should be a high level summary - aim for 3-5 concise bullet points.
        - Second section should be a list of topics that are discussed.
        - Third section should be a list of key learnings or takeaways.
        - Fourth section should be to flag any other interesting information that is not covered in the other sections, and / or perhpas one or a few meanigful quotes from any of the speakers.

        Keep it short and concise, each section should be well apparent and contain up to 10 bullet points.
        
        Text: {text}
        
        Education Summary:"""
    )
    
    chain = summarization_prompt | llm
    response = chain.invoke({"text": state["text"]})
    return {"summary": response.content}

def create_graph():
    """Create and configure the processing workflow graph."""
    # Initialize the graph
    workflow = StateGraph(State)

    # Add nodes to the graph
    workflow.add_node("classify", classification_node)
    # workflow.add_node("human_entity_extraction", human_entity_extraction_node)
    workflow.add_node("company_entity_extraction", company_entity_extraction_node)
    workflow.add_node("summarize_finance", summarize_finance_node)
    workflow.add_node("summarize_news", summarize_news_node)
    workflow.add_node("summarize_podcast", summarize_podcast_node)
    workflow.add_node("summarize_education", summarize_education_node)

    # Define the conditional logic for routing based on classification
    def router(state: State):
        if state["classification"] == "finance":
            return "summarize_finance"
        elif state["classification"] == "news":
            return "summarize_news"
        elif state["classification"] == "podcast":
            return "summarize_podcast"
        elif state["classification"] == "education":
            return "summarize_education"
        else:
            return "summarize_podcast"  # Default to podcast summarization for other categories

    # Configure the graph edges
    workflow.set_entry_point("classify")  # Set the entry point
    # workflow.add_edge("classify", "human_entity_extraction")
    # workflow.add_edge("human_entity_extraction", "company_entity_extraction")
    workflow.add_edge("classify", "company_entity_extraction")
    
    # Add conditional edge from company_entity_extraction to summarization nodes
    workflow.add_conditional_edges(
        "company_entity_extraction",
        router,
        {
            "summarize_finance": "summarize_finance",
            "summarize_news": "summarize_news",
            "summarize_podcast": "summarize_podcast",
            "summarize_education": "summarize_education"
        }
    )
    
    # Connect summarization nodes to end
    workflow.add_edge("summarize_finance", END)
    workflow.add_edge("summarize_news", END)
    workflow.add_edge("summarize_podcast", END)
    workflow.add_edge("summarize_education", END)

    # Compile the graph
    return workflow.compile()

def process_transcript(text: str):
    """
    Process a transcript through the entire pipeline.
    
    Args:
        text (str): The transcript text to process
        
    Returns:
        dict: The final state containing all processing results
    """
    logger.info("Starting transcript processing pipeline...")
    logger.info(f"Input text length: {len(text)} characters")
    
    # Create the workflow graph
    graph = create_graph()
    
    # Initialize the state with the input text
    initial_state = {"text": text}
    
    # Run the workflow
    logger.info("Running workflow...")
    result = graph.invoke(initial_state)
    logger.info("Workflow completed successfully")
    return result

if __name__ == "__main__":
    logger.info("Loading transcript data...")
    with open('./scripts/data/transcript2.json', 'r') as f:
        transcript_data = json.load(f)
    sample_text = str(transcript_data)
    logger.info("Transcript data loaded successfully")
    
    result = process_transcript(sample_text)
    logger.info("\n=== Final Results ===")
    logger.info(f"Classification: {result['classification']}")
    logger.info(f"Human Entities: {result['human_entities']}")
    logger.info(f"Company Entities: {result['company_entities']}")
    logger.info("\nSummary:")
    print(result["summary"]) 