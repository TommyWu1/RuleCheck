from pydantic import BaseModel, ConfigDict, Field


class GeneratedTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3)
    months_employed: int = Field(ge=0)
    expected_days: int = Field(ge=0)
    rationale: str = Field(min_length=3)


class GeneratedSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: list[GeneratedTest] = Field(min_length=5, max_length=5)
