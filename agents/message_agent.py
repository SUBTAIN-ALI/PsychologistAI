from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config.settings import settings
from typing import Dict, Any

def get_message_agent_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.4
    )

def generate_outreach_message(lead_data: Dict[str, Any]) -> str:
    """
    Generates a personalized, supportive outreach message for a qualified lead.
    """
    llm = get_message_agent_llm()
    
    prompt = PromptTemplate.from_template(
        "You are representing a clinical psychologist. Write a personalized, supportive, "
        "and empathetic outreach message to someone who recently posted on {platform}.\n\n"
        "Lead Name: {name}\n"
        "Their Post: {post}\n"
        "Topic of Interest: {interest}\n\n"
        "Requirements:\n"
        "- Maintain a supportive and empathetic tone.\n"
        "- Professional communication style.\n"
        "- Reference the context of their post gently.\n"
        "- Avoid aggressive marketing or sales language.\n"
        "- Offer to share strategies or have a brief consultation regarding {interest}.\n"
        "- DO NOT diagnose them.\n"
        "- Keep it concise (2-3 paragraphs max).\n\n"
        "Message:"
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "name": lead_data.get("name"),
        "platform": lead_data.get("platform"),
        "post": lead_data.get("post"),
        "interest": lead_data.get("interest")
    })
    
    return response.content.strip()
