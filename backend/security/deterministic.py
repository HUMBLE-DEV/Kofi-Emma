from langchain_groq import ChatGroq



def is_research_document_heuristic(text: str) -> tuple[bool, str]:
    """Uses regex and heuristics to check if the text looks like an academic paper."""
    text_lower = text.lower()
    
    # 1. Length check: Academic papers are usually > 2000 characters
    if len(text) < 2000:
        return False, "Document is too short to be a research paper."

    # 2. Keyword check: Look for academic structural markers
    academic_markers = [
        "abstract", "introduction", "methodology", "results", 
        "discussion", "conclusion", "references", "et al.", 
        "doi", "figure", "table", "dataset"
    ]
    
    # Count how many markers appear
    matches = sum(1 for marker in academic_markers if marker in text_lower)
    
    # 3. The Threshold: If it has at least 4 of these markers, it's probably a paper
    if matches >= 4:
        return True, "Valid research document detected."
    else:
        return False, "This document does not look like a standard academic paper (missing abstract, methodology, references, etc.)."




# def is_research_document_ai(text: str) -> tuple[bool, str]:
#     # Only send the first 1500 characters to save tokens
#     sample_text = text[:1500]
    
#     # Use the smallest, fastest model
#     llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=20)
    
#     prompt = f"""Is this text from an academic research paper or a dataset? 
#     Reply with exactly 'YES' or 'NO'.
    
#     Text: {sample_text}"""
    
#     try:
#         response = llm.invoke(prompt)
#         if "YES" in response.content.upper():
#             return True, "Valid"
#         else:
#             return False, "The AI determined this is not an academic document."
#     except Exception:
#         return True, "Check bypassed."