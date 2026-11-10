from backend.guardrails import get_greeting_response


def test_greetings():
    assert get_greeting_response("hello") is not None
    assert get_greeting_response("Hi there!") is not None
    assert get_greeting_response("Hello, write a literature review") is None # Safely ignores the greeting if there's a real task!