import enum

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class AgentState(MessagesState):
    """Graph state: the message log plus the research routing fields.

    Only annotations here. (Defaults written in a TypedDict body are NOT applied by
    LangGraph, so the code always reads these keys with .get(key, default).)
    """

    route: str
    thread_id: str
    research_topic: str
    research_paper: str
    study_snapshot: str
    citation_chain: str
    research_gaps: str
    literature_review: str
    awaiting_project: bool


class CategoryEnum(str, enum.Enum):
    study_snapshot = "study_snapshot"
    citation_chainer = "citation_chainer"
    research_gaps_matrix = "research_gaps_matrix"
    literature_review_writer = "literature_review_writer"
    chat = "chat"


class IntentClassification(BaseModel):
    category: CategoryEnum = Field(description="The classified category based on the user's intent.")
    topic: str = Field(
        default="",
        description=(
            "The core research topic as a concise search-engine query. Remove conversational filler "
            "(e.g. 'help me with', 'for my project'). Example: 'Write a review on AI for nursing' "
            "becomes 'Artificial Intelligence Nursing'."
        ),
    )
    paper_identifier: str = Field(
        default="",
        description="A DOI or the exact paper title for citation_chainer, otherwise an empty string.",
    )





class StandardChatRequest(BaseModel):
    message: str
    thread_id: str
    model: str = "gemini-2.5-flash"


