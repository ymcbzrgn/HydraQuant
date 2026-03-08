import os
import logging
from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END

from hybrid_retriever import HybridRetriever

# Load Env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
logger = logging.getLogger(__name__)

# LLM Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env")

# We use gemini-2.5-flash for almost all cognitive tasks due to speed/cost ratio
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", api_key=GEMINI_API_KEY, temperature=0, convert_system_message_to_human=True)
web_search_tool = DuckDuckGoSearchRun()

# Single persistent retriever instance
retriever = HybridRetriever()

# --- Graph State Definition ---
class GraphState(TypedDict):
    """
    State dictionary for the LangGraph RAG Agent.
    """
    question: str
    generation: str
    web_search: str # "yes" or "no"
    documents: List[str]

# --- Nodes (Modular Functions) ---

def retrieve(state: GraphState):
    """Retrieves documents from the Hybrid Search Vector DB."""
    logger.info("---RETRIEVE---")
    question = state["question"]
    
    # Get top 5 from hybrid retriever
    results = retriever.search(question, top_k=5)
    documents = [res["text"] for res in results]
    
    return {"documents": documents, "question": question}

def generate(state: GraphState):
    """Generates the final answer using retrieved documents."""
    logger.info("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    context = "\n\n".join(documents)
    prompt = f"""You are Freqbot, an expert crypto market analyst. 
Use the following pieces of retrieved context to answer the user's question. 
If you don't know the answer, just say that you don't know. 
Keep the answer concise and geared towards financial reasoning.

Context: 
{context}

Question: {question}
Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"documents": documents, "question": question, "generation": response.content}

def grade_documents(state: GraphState):
    """
    CRAG Mechanism: Discriminator node.
    Evaluates if documents are relevant to the question.
    """
    logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Prompt for relevance grading (outputs structured YES/NO ideally, simplified here)
    system_msg = "You are a grader assessing relevance of a retrieved document to a user question. " \
                 "If the document contains keywords or semantic meaning related to the question, grade it as 'yes'. " \
                 "Otherwise, grade it as 'no'. Give ONLY a 'yes' or 'no' string as your response without explanation."
                 
    filtered_docs = []
    web_search = "no"
    
    for d in documents:
        prompt = f"Retrieved document: \n\n {d} \n\n User question: {question} \n\n Is it relevant? (yes/no):"
        response = llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=prompt)
        ])
        
        grade = response.content.strip().lower()
        if "yes" in grade:
            filtered_docs.append(d)
        else:
            logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "yes"
            continue
            
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def web_search(state: GraphState):
    """Web search based fallback (CRAG execution)"""
    logger.info("---WEB SEARCH FALLBACK---")
    question = state["question"]
    documents = state["documents"]
    
    try:
        docs = web_search_tool.invoke(question)
        documents.append(docs)
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        
    return {"documents": documents, "question": question}

# --- Conditional Edges ---

def decide_to_generate(state: GraphState) -> Literal["generate", "web_search"]:
    """Determines whether to execute web search or generate immediately."""
    logger.info("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]
    
    if web_search == "yes":
        logger.info("---DECISION: ALL/SOME DOCS IRRELEVANT, ROUTE TO WEB SEARCH---")
        return "web_search"
    else:
        logger.info("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state: GraphState) -> Literal["useful", "not_useful", "not_supported"]:
    """
    Self-RAG Mechanism: Self-Critique.
    Validates Hallucination (against docs) and Answer Relevance (against question).
    """
    logger.info("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    # Hallucination Check
    system_msg_hall = "You are a grader checking for hallucinations. Does the following generation purely rely on the provided documents? " \
                      "Provide ONLY 'yes' or 'no'."
                      
    context = "\n\n".join(documents)
    prompt_hall = f"Documents: {context} \n\n Generation: {generation} \n\n Is the generation grounded in the documents? (yes/no):"
    
    res_hall = llm.invoke([SystemMessage(content=system_msg_hall), HumanMessage(content=prompt_hall)])
    hallucination_grade = res_hall.content.strip().lower()
    
    if "yes" in hallucination_grade:
        logger.info("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Answer Relevance Check
        logger.info("---GRADE GENERATION vs QUESTION---")
        system_msg_rel = "You are a grader assessing if an answer resolves a question. Answer ONLY 'yes' or 'no'."
        prompt_rel = f"Question: {question} \n\n Answer: {generation} \n\n Does it address the question? (yes/no):"
        
        res_rel = llm.invoke([SystemMessage(content=system_msg_rel), HumanMessage(content=prompt_rel)])
        relevance_grade = res_rel.content.strip().lower()
        
        if "yes" in relevance_grade:
            logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not_useful"
    else:
        logger.info("---DECISION: GENERATION IS HALLUCINATED---")
        return "not_supported"

# --- Graph Construction ---
workflow = StateGraph(GraphState)

# Define nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search)

# Define edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate",
    }
)

workflow.add_edge("web_search", "generate")

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not_supported": "generate", # Generate again (retry without hallucination)
        "useful": END,               # Finished successfully
        "not_useful": "web_search",  # Not answering question, get more info from web
    }
)

# Compile graph
rag_bot = workflow.compile()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Initiating Phase 3.1 Advanced RAG Flow Test...")
    
    test_query = "What exactly is happening with Tether reserves today?"
    inputs = {"question": test_query}
    
    # Execute the Stateful Graph
    final_output = None
    for output in rag_bot.stream(inputs):
        for key, value in output.items():
            logger.info(f"Finished Node: {key}")
            final_output = value
            
    print("\n\n=== FINAL ANSWER ===")
    print(final_output["generation"])
