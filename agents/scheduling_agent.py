from typing import Dict, Any
from tools.calendar_tool import calendar_tool

def handle_scheduling(lead_name: str, requested_time: str = None) -> Dict[str, Any]:
    """
    Agent logic to handle scheduling. If no time is requested, returns available slots.
    If a time is requested, attempts to book it.
    """
    if not requested_time:
        slots = calendar_tool.get_available_slots()
        return {
            "status": "requires_selection",
            "message": "Please choose from the following slots:",
            "slots": slots
        }
    else:
        result = calendar_tool.schedule_appointment(lead_name, requested_time)
        return result
