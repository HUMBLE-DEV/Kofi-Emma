class PermanentRejection(ValueError):
    """The document will never work (wrong type, no text, not a research paper).

    The message is shown to the student, so write it in plain words.
    Anything else that goes wrong is treated as temporary and retried.
    """
