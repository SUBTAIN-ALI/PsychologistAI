from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from workflows.research_workflow import run_research_query
from config.settings import settings
from typing import List, Dict, Any

def get_conversation_agent_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.3
    )

def handle_conversation(messages: List[Any], use_rag: bool = False) -> str:
    """
    Handles a multi-turn conversation. Optionally integrates knowledge from RAG.
    """
    llm = get_conversation_agent_llm()
    
    system_prompt = (
        "You are an empathetic, professional AI assistant for a psychology practice. "
        "Your goal is to answer questions related to emotional well-being, stress, and burnout. "
        "Provide helpful insights while maintaining context. "
        "Gently guide the user towards scheduling a consultation if they seem receptive. "
        "DO NOT provide medical diagnoses."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    # If the user asks a specific question that could benefit from research,
    # we can inject RAG context into the latest message.
    # For a real system, an agent with tools (function calling) is better, 
    # but for this script we do a manual check or just pass it in.
    
    if use_rag and len(messages) > 0 and isinstance(messages[-1], HumanMessage):
        latest_query = messages[-1].content
        # In a real system you might have an LLM decide whether to query RAG.
        # Here we do it if requested.
        try:
            rag_context = run_research_query(latest_query)
            augmented_prompt = f"User asked: {latest_query}\n\nHere is some research context you can optionally use to inform your empathetic answer:\n{rag_context}"
            messages[-1] = HumanMessage(content=augmented_prompt)
        except Exception as e:
            print(f"RAG query failed, proceeding without context. Error: {e}")

    chain = prompt | llm
    response = chain.invoke({"messages": messages})
    
    return response.content
