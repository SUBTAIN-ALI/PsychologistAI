from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config.settings import settings

def get_manager_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.0
    )

def evaluate_conversation(state: Dict[str, Any]) -> str:
    """
    Evaluates where the workflow should go next based on the latest messages.
    Returns the name of the next node.
    """
    messages = state.get("messages", [])
    if not messages:
        # If no messages, need to qualify the lead
        return "qualify_lead"
    
    # Manager decides if the conversation indicates the user wants to schedule,
    # or if the conversation should continue, or if it should end.
    llm = get_manager_llm()
    
    prompt = PromptTemplate.from_template(
        "You are a Manager Agent determining the next step in a lead workflow.\n\n"
        "Recent Conversation:\n{chat_history}\n\n"
        "Options:\n"
        "- 'schedule': User expresses clear intent to book a consultation or requests times.\n"
        "- 'continue': User is asking questions and the conversation is active.\n"
        "- 'end': User explicitly refused, or the conversation naturally ended.\n\n"
        "Analyze the last messages and choose strictly ONE option from above:\n"
        "Decision (schedule/continue/end):"
    )
    
    # Formatting for context
    chat_str = "\n".join([f"{m.type}: {m.content}" for m in messages[-3:]])
    
    chain = prompt | llm
    response = chain.invoke({"chat_history": chat_str})
    decision = response.content.strip().lower()
    
    if "schedule" in decision:
        return "scheduler"
    elif "end" in decision:
        return "end"
    else:
        # Default to continuous conversation if not explicitly schedule or end
        return "chat"
