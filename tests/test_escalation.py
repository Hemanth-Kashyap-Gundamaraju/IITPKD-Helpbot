"""
Description: Unit tests for backend.utils.escalation.EscalationHandler.
"""
from backend.utils.escalation import EscalationHandler

def test_escalation_handler():
    handler = EscalationHandler()
    
    # Test fee issue
    response = handler.check_escalation("my fee is not reflecting in the portal")
    assert response is not None
    assert "accounts@iitpkd.ac.in" in response

    # Test hostel issue
    response = handler.check_escalation("i want to change my hostel room")
    assert response is not None
    assert "hostelmanager@iitpkd.ac.in" in response

    # Test email login
    response = handler.check_escalation("smail login not working")
    assert response is not None
    assert "sysadmin@iitpkd.ac.in" in response
    
    # Test ragging
    response = handler.check_escalation("someone is ragging me in hostel")
    assert response is not None
    assert "antiragging" in response
    
    # Test non-escalation query
    response = handler.check_escalation("who is the director of iitpkd?")
    assert response is None
