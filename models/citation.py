from typing import List

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """
    A citation for a research answer.
    """

    title: str = Field(description="Title of the research paper.")
    authors: List[str] = Field(description="List of authors of the research paper.")
    year: int = Field(description="Year of publication.")
    doi: str = Field(description="Digital Object Identifier of the research paper.")
    url: str = Field(description="URL to access the research paper.")
    page_numbers: int = Field(
        description="Page numbers referenced in the research paper."
    )
