from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from rag.retriever import get_retriever
from config.settings import settings

def get_research_assistant_chain():
    """
    Returns a RetrievalQA chain for the research assistant.
    """
    llm = ChatOpenAI(
        model=settings.LLM_MODEL, 
        api_key=settings.OPENAI_API_KEY,
        temperature=0.2
    )
    
    retriever = get_retriever()

    # Define the system prompt
    system_prompt = (
        "You are an expert AI Research Assistant for a clinical psychologist. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Provide scientific, evidence-based responses where possible. "
        "Maintain a professional, objective tone suited for a mental health professional.\n\n"
        "Context:\n{context}"
    )

    prompt = PromptTemplate.from_template(
        "System: " + system_prompt + "\n\nUser: {input}\nAssistant:"
    )

    # Create the stuff documents chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the retrieval chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def run_research_query(query: str) -> str:
    """
    Executes a query against the knowledge base and returns the answer.
    """
    chain = get_research_assistant_chain()
    response = chain.invoke({"input": query})
    return response["answer"]
