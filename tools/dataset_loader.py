import json
import os
from typing import List, Dict, Any

def load_synthetic_leads(filepath: str = "data/synthetic_leads.json") -> List[Dict[str, Any]]:
    """
    Loads the synthetic leads from the specified JSON file.
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_path, filepath)
    
    try:
        with open(full_path, "r") as f:
            leads = json.load(f)
        return leads
    except FileNotFoundError:
        print(f"Error: Could not find lead data at {full_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {full_path}")
        return []

def get_new_leads() -> List[Dict[str, Any]]:
    """
    Retrieves leads that have the status 'new'.
    """
    leads = load_synthetic_leads()
    return [lead for lead in leads if lead.get("status") == "new"]

if __name__ == "__main__":
    leads = get_new_leads()
    print(f"Loaded {len(leads)} new leads.")
    if leads:
        print("First lead:", leads[0])
