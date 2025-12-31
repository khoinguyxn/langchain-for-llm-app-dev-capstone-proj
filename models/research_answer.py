from typing import List

from pydantic import BaseModel, Field

from models.citation import Citation


class ResearchAnswer(BaseModel):
    """
    A research answer generated from multiple research papers.
    """

    answer: str = Field(description="The generated research answer.")
    citations: List[Citation] = Field(
        description="List of citations supporting the research answer."
    )
    confidence_score: float = Field(description="Confidence level from 0 to 1.")
