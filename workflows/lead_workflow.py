import operator
from typing import Annotated, Sequence, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END

from agents.lead_generation_agent import qualify_lead
from agents.message_agent import generate_outreach_message
from agents.conversation_agent import handle_conversation
from agents.scheduling_agent import handle_scheduling
from agents.manager_agent import evaluate_conversation

# Define State
class LeadState(TypedDict):
    # Core Data
    lead_id: int
    lead_data: dict
    status: str # 'new', 'qualified', 'unqualified', 'contacted', 'scheduled', 'ended'
    
    # Workflow Data
    messages: Annotated[Sequence[BaseMessage], operator.add]
    outreach_message: str
    schedule_data: dict

def node_qualify_lead(state: LeadState) -> LeadState:
    """Classifies if the lead should be contacted."""
    print(f"[Lead Gen Agent] Qualifying lead {state['lead_data'].get('name', '')}...")
    qualification = qualify_lead(state["lead_data"])
    return {"status": qualification}

def route_after_qualification(state: LeadState):
    """Router determining path after qualification"""
    if state["status"] == "qualified":
        return "generate_outreach"
    else:
        return END

def node_generate_outreach(state: LeadState) -> LeadState:
    """Generates the initial supportive message to send."""
    print("[Message Agent] Generating supportive outreach message...")
    message = generate_outreach_message(state["lead_data"])
    ai_message = AIMessage(content=message)
    return {
        "outreach_message": message,
        "messages": [ai_message],
        "status": "contacted"
    }
    
def node_manager_routing(state: LeadState):
    """Manager agent determines next step based on state/messages"""
    # Simply delegates decision to manager_agent
    decision = evaluate_conversation(state)
    print(f"[Manager Agent] Routing decision: {decision}")
    
    if decision == "scheduler":
        return "scheduler"
    elif decision == "chat":
        return "chat"
    else:
        return END

def node_chat(state: LeadState) -> LeadState:
    """Handles an incoming user message via conversation agent with RAG."""
    print("[Conversation Agent] Replying to conversation...")
    # Assume the latest message in state["messages"] is a human message
    response = handle_conversation(state["messages"], use_rag=True)
    ai_message = AIMessage(content=response)
    
    return {
        "messages": [ai_message]
    }

def node_scheduler(state: LeadState) -> LeadState:
    """Attempts to handle scheduling logic."""
    print("[Scheduling Agent] Handling scheduling options...")
    
    lead_name = state["lead_data"].get("name", "Unknown Lead")
    
    # See if user provided a time recently in chat (simplified mock logic)
    # A real system would use a tool parser. 
    # For now, let's just ask for available slots.
    schedule_info = handle_scheduling(lead_name)
    
    # Add response to messages
    if schedule_info["status"] == "requires_selection":
        slots_str = "\\n".join(schedule_info["slots"])
        msg = f"{schedule_info['message']}\\n{slots_str}"
        return {"messages": [AIMessage(content=msg)]}
    else:
        return {
            "messages": [AIMessage(content=schedule_info["message"])],
            "schedule_data": schedule_info,
            "status": "scheduled"
        }

# Build Graph
builder = StateGraph(LeadState)

# Add Nodes
builder.add_node("qualify_lead", node_qualify_lead)
builder.add_node("generate_outreach", node_generate_outreach)
builder.add_node("chat", node_chat)
builder.add_node("scheduler", node_scheduler)

# Add Edges
builder.add_edge(START, "qualify_lead")
builder.add_conditional_edges("qualify_lead", route_after_qualification)
builder.add_edge("generate_outreach", END) # The system waits for user reply before chatting again

# Conversational Loop Edges (Triggered usually via an endpoint where current state is resumed)
builder.add_conditional_edges("chat", node_manager_routing)
# When scheduler finishes suggesting options, we stop for user.
builder.add_edge("scheduler", END) 

def get_lead_workflow():
    return builder.compile()

if __name__ == "__main__":
    from tools.dataset_loader import get_new_leads
    leads = get_new_leads()
    if leads:
        workflow = get_lead_workflow()
        initial_state = {
            "lead_id": leads[0]["lead_id"],
            "lead_data": leads[0],
            "status": "new",
            "messages": [],
            "outreach_message": "",
            "schedule_data": {}
        }
        res = workflow.invoke(initial_state)
        import pprint
        pprint.pprint(res)
