from typing import List, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """
    A citation for a research answer.
    """

    title: str = Field(description="Title of the research paper.")
    authors: List[str] = Field(
        default_factory=list, description="List of authors of the research paper."
    )
    year: Optional[int] = Field(
        default=None, description="Year of publication (if available)."
    )
    doi: Optional[str] = Field(
        default=None,
        description="Digital Object Identifier (DOI) of the research paper.",
    )
    url: Optional[str] = Field(default=None, description="URL to access the research paper.")
    page_numbers: Optional[int] = Field(
        default=None,
        description="Page numbers referenced in the research paper (if available).",
    )
