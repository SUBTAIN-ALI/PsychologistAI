from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from workflows.lead_workflow import get_lead_workflow
from workflows.research_workflow import run_research_query
from tools.dataset_loader import get_new_leads
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage

router = APIRouter()

# Global state to keep track of conversations in memory (for demo)
# In production, this would be a database.
active_sessions = {}
lead_workflow = get_lead_workflow()

class ChatRequest(BaseModel):
    lead_id: int
    message: str

class ResearchQuery(BaseModel):
    query: str

@router.post("/visibility/process_new_leads")
def process_leads():
    """
    Simulates fetching new leads from the synthetic dataset and passing them through the graph
    to the point of generating the initial outreach message.
    """
    leads = get_new_leads()
    results = []
    
    for lead in leads:
        lead_id = lead["lead_id"]
        # In a real app check if lead is already processed
        if lead_id not in active_sessions:
            initial_state = {
                "lead_id": lead_id,
                "lead_data": lead,
                "status": "new",
                "messages": [],
                "outreach_message": "",
                "schedule_data": {}
            }
            # Start workflow
            try:
                state = lead_workflow.invoke(initial_state)
                active_sessions[lead_id] = state
                results.append({
                    "lead_id": lead_id,
                    "name": lead["name"],
                    "status": state.get("status"),
                    "outreach_message": state.get("outreach_message")
                })
            except Exception as e:
                print(f"Error processing lead {lead_id}: {e}")
                results.append({"lead_id": lead_id, "error": str(e)})

    return {"processed_leads": results}

@router.post("/visibility/chat")
def chat_with_lead(req: ChatRequest):
    """
    Continues a conversation with a lead. The workflow decides how to respond and route.
    """
    if req.lead_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Lead session not found.")
        
    state = active_sessions[req.lead_id]
    
    # Append the user's message
    messages = list(state.get("messages", []))
    messages.append(HumanMessage(content=req.message))
    state["messages"] = messages
    
    # Resume the workflow by invoking it from the chat node
    # Since we use conditional edges, invoking the graph starting at the 'manager routing' 
    # would be technically correct, but to keep it simple we just invoke the full state.
    # In a proper LangGraph persisting setup we'd resume the graph execution.
    # For this demo, we'll manually route the updated state through manager logic or just call the chat node.
    
    # Let's run it from the chat node manually to simulate resuming.
    from agents.manager_agent import evaluate_conversation
    decision = evaluate_conversation(state)
    
    try:
        if decision == "scheduler":
            from workflows.lead_workflow import node_scheduler
            new_state = node_scheduler(state)
            state.update(new_state)
        elif decision == "chat":
            from workflows.lead_workflow import node_chat
            new_state = node_chat(state)
            state.update(new_state)
        else: # end
            state["status"] = "ended"
            
        active_sessions[req.lead_id] = state # Update memory
        
        # Get AI's latest message
        latest_msgs = state.get("messages", [])
        response_msg = latest_msgs[-1].content if latest_msgs else "Conversation ended."
        
        return {
            "lead_id": req.lead_id,
            "status": state.get("status"),
            "response": response_msg
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/research/query")
def ask_research_assistant(req: ResearchQuery):
    """
    Queries the RAG system for psychology research.
    """
    try:
        answer = run_research_query(req.query)
        return {"query": req.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
