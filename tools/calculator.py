import math
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


class CalcInput(BaseModel):
    expression: str = Field(description="A math expression e.g. '191.6 - 130.0'")


def calculate(expression: str) -> str:
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "max": max, "min": min})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{result}"
    except Exception as e:
        return f"Error: {e}"


calculator_tool = StructuredTool.from_function(
    func=calculate,
    name="calculator",
    description="Evaluate math expressions. Use for strike rate differences, economy comparisons, run rate calculations etc.",
    args_schema=CalcInput,
)