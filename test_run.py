import os
import sys

# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.dataset_loader import get_new_leads
from workflows.lead_workflow import get_lead_workflow
from langchain_core.messages import HumanMessage
import pprint

def run_test():
    print("Fetching leads...")
    leads = get_new_leads()
    if not leads:
        print("No leads found.")
        return

    lead = leads[0]
    print(f"Testing workflow with lead: {lead['name']} ({lead['interest']})")

    workflow = get_lead_workflow()

    initial_state = {
        "lead_id": lead["lead_id"],
        "lead_data": lead,
        "status": "new",
        "messages": [],
        "outreach_message": "",
        "schedule_data": {}
    }

    print("\n--- INITIATING WORKFLOW ---")
    
    # We do a mock invocation and print the state output
    # By default, the graph will run: 
    # qualify_lead -> (if qualified) -> generate_outreach -> END
    state = workflow.invoke(initial_state)
    
    print("\n--- STATE AFTER INITIAL RUN ---")
    print(f"Status: {state['status']}")
    print(f"Outreach Message:\n{state.get('outreach_message')}")
    
    # Simulate user sending a message back
    if state["status"] == "contacted":
        print("\n--- SIMULATING USER REPLY ---")
        user_msg = HumanMessage(content="Thank you. Yes, I would like to hear some strategies.")
        
        # Append message
        state["messages"].append(user_msg)
        
        # Since we hit END after generate_outreach, to continue we can directly call node_chat 
        # or use the manager to decide. Let's use the manager routing.
        from agents.manager_agent import evaluate_conversation
        decision = evaluate_conversation(state)
        print(f"\nManager Decision: {decision}")
        
        if decision == "chat":
            from workflows.lead_workflow import node_chat
            print("\nCalling Chat Agent...")
            new_state_data = node_chat(state)
            state.update(new_state_data)
            print(f"Chat Agent Reply: {state['messages'][-1].content}")
            
if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is available in the environment
    import os
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key":
        print("Test script ready. Please provide a valid OpenAI API Key in your .env file to run successfully.")
    else:
        run_test()
