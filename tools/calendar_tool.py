from datetime import datetime, timedelta
import random

class CalendarTool:
    """
    A mock tool to simulate checking calendar availability and scheduling appointments.
    """
    def __init__(self):
        # Generate some mock available slots starting from tomorrow
        self.available_slots = self._generate_mock_slots()
        self.appointments = []
        
    def _generate_mock_slots(self):
        slots = []
        now = datetime.now()
        for i in range(1, 5): # Next 4 days
            date = now + timedelta(days=i)
            # Add random hour slots between 9 AM and 4 PM
            for hour in random.sample(range(9, 16), 3):
                slot = date.replace(hour=hour, minute=0, second=0, microsecond=0)
                slots.append(slot.strftime("%Y-%m-%d %H:%M"))
        return sorted(slots)

    def get_available_slots(self, num_slots: int = 3) -> list[str]:
        """
        Returns a list of available time slots.
        """
        return self.available_slots[:num_slots]

    def schedule_appointment(self, lead_name: str, preferred_time: str) -> dict:
        """
        Simulates scheduling an appointment.
        """
        if preferred_time in self.available_slots:
            self.available_slots.remove(preferred_time)
            appointment = {
                "lead_name": lead_name,
                "time": preferred_time,
                "status": "confirmed"
            }
            self.appointments.append(appointment)
            return {"status": "success", "message": f"Appointment confirmed for {lead_name} at {preferred_time}", "details": appointment}
        else:
            return {"status": "error", "message": f"Time slot {preferred_time} is not available."}

calendar_tool = CalendarTool()
