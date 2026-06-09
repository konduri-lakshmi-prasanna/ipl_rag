from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


class CorpusInput(BaseModel):
    query: str = Field(description="The query to check against corpus boundaries")


def corpus_check(query: str) -> str:
    return (
        "The IPL Intelligence dataset covers: team profiles (10 teams, 2024 season), "
        "batting stats (20 players), bowling stats (15 players), "
        "head-to-head records (10 matchups), venue/pitch reports (8 venues), "
        "season-wise performance (2019-2024), recent form (last 5 matches, 2024), "
        "IPL records and milestones. "
        "NOT included: auction prices, player salaries, ICC rankings, "
        "T20 World Cup data, or match schedules."
    )


corpus_tool = StructuredTool.from_function(
    func=corpus_check,
    name="corpus_boundary_check",
    description="Check what is and isn't available in the IPL dataset.",
    args_schema=CorpusInput,
)