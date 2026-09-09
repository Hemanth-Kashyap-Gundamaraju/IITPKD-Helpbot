"""
Description: Utility to detect and handle "escalation" queries — questions about
    personal issues (fees, hostels, mailing lists) that the bot cannot answer
    from the public website. Provides direct contact info for the relevant
    institute offices instead of querying the LLM.
Inputs: none (defines a class and functions).
Outputs: none.
Dependencies: re.
Utilities: used by chain_builder.py.
"""
import re

class EscalationHandler:
    """
    Description: Checks if a user's question matches known patterns for
        personal issues that require human intervention.
    """

    def __init__(self):
        # A dictionary mapping regex patterns to the appropriate response/contact
        self.escalation_rules = {
            r"\b(fee|payment|paid|receipt).*(not reflecting|issue|problem|failed)\b": (
                "For fee payment issues, please contact the Accounts Section at accounts@iitpkd.ac.in or call 04923-226500."
            ),
            r"(?=.*\b(hostel|room)\b)(?=.*\b(not allotted|change|issue)\b)": (
                "For hostel allotment or related issues, please contact the Hostel Office at hostelmanager@iitpkd.ac.in."
            ),
            r"\b(mailing list|email|smail|login).*(not working|added|password)\b": (
                "For IT and email-related issues, please contact the Central IT Facilities (CITF) team at sysadmin@iitpkd.ac.in."
            ),
            r"\b(medical|health|hospital|doctor)\b": (
                "For medical emergencies or health center queries, please contact the Institute Health Center at medical@iitpkd.ac.in or call the emergency number."
            ),
            r"\b(ragging|harassment)\b": (
                "If you are reporting an incident of ragging or harassment, please immediately contact the Anti-Ragging Squad at antiragging@iitpkd.ac.in or call the national helpline 1800-180-5522."
            )
        }

        # Pre-compile regexes for performance
        self.compiled_rules = {
            re.compile(pattern, re.IGNORECASE): response
            for pattern, response in self.escalation_rules.items()
        }

    def check_escalation(self, question):
        """
        Description: Checks a question against all escalation rules.
        Inputs: question (string).
        Outputs: string (the escalation message) if matched, None otherwise.
        """
        for pattern, response in self.compiled_rules.items():
            if pattern.search(question):
                return response
        return None

# Singleton instance
_escalation_handler = EscalationHandler()

def get_escalation_handler():
    return _escalation_handler
