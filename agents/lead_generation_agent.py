from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config.settings import settings
from typing import Dict, Any

def get_lead_qualification_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.1
    )

def qualify_lead(lead_data: Dict[str, Any]) -> str:
    """
    Analyzes a lead's post to determine if they might benefit from therapy, based on safe criteria.
    Returns 'qualified' or 'unqualified'.
    """
    llm = get_lead_qualification_llm()
    
    prompt = PromptTemplate.from_template(
        "You are an AI assistant for a psychology practice. Your job is to analyze social media posts "
        "and determine if the author might benefit from mental health support for issues like burnout, "
        "stress, anxiety, or career pressure.\n\n"
        "Lead Info:\n"
        "Name: {name}\n"
        "Platform: {platform}\n"
        "Post: {post}\n\n"
        "Analyze the post carefully. Do NOT provide a medical diagnosis. "
        "If the person is expressing distress, stress, anxiety, or burnout that could "
        "be helped by talking to a professional, respond with exactly: qualified\n"
        "Otherwise, respond with exactly: unqualified\n\n"
        "Classification:"
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "name": lead_data.get("name"),
        "platform": lead_data.get("platform"),
        "post": lead_data.get("post")
    })
    
    result = response.content.strip().lower()
    return "qualified" if "qualified" in result and "unqualified" not in result else "unqualified"
