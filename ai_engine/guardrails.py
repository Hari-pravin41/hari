import re
from typing import Tuple

class SafetyGuard:
    """
    Enforces security and safety policies on Input and Output.
    Prevents injection attacks and PII leakage.
    """

    def __init__(self):
        # Basic patterns for PII (Mock)
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b', # SSN-like
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b' # Email
        ]
        
    def validate_input(self, text: str) -> Tuple[bool, str]:
        """
        Checks input for huge size or injection attempts.
        Returns: (is_safe, error_message)
        """
        if len(text) > 10000:
            return False, "Input exceeds maximum length (10k chars)."
            
        # Placeholder for prompt injection detection
        if "<script>" in text.lower():
            return False, "Malicious content detected."
            
        return True, ""

    def sanitize_output(self, text: str) -> str:
        """
        Redacts sensitive chunks from the model output before sending to UI.
        """
        sanitized = text
        for pattern in self.pii_patterns:
            sanitized = re.sub(pattern, "[REDACTED]", sanitized)
        return sanitized

if __name__ == "__main__":
    guard = SafetyGuard()
    valid, msg = guard.validate_input("Hello user@example.com")
    print(f"Valid: {valid}, Msg: {msg}")
    print(f"Sanitized: {guard.sanitize_output('Contact me at user@example.com for info.')}")
