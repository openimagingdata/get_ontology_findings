import csv
import os
from typing import Literal, Optional

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

RADLEX_FILE_NAME = "data/radlex_410.csv"


class RadLexTerm(BaseModel):
    id: str = Field(description="The unique identifier for the RadLex term.", pattern=r"^RID\d+$")
    term: str
    synonyms: Optional[list[str]] = Field(default=None, description="A list of synonyms for the RadLex term.")
    definition: Optional[str] = None

    def pretty_str(self, include_synonyms: bool = False) -> str:
        """Return a pretty string representation of the RadLex term."""
        out = f"{self.id}: {self.term}"
        if include_synonyms and self.synonyms:
            out += f" [{', '.join(self.synonyms)}]"
        return out


def load_radlex_terms() -> list[RadLexTerm]:
    """Load RadLex terms from a CSV file."""
    # Placeholder for actual loading logic
    # This should read the CSV file and return a list of dictionaries

    def clean_radlex_term(term: dict[str, str]) -> RadLexTerm:
        out: dict[str, str | list[str]] = {
            "id": term["id"],
            "term": term["term"],
        }
        if term["synonyms"]:
            out["synonyms"] = term["synonyms"].split("|")
        if term["definition"]:
            out["definition"] = term["definition"]
        return RadLexTerm.model_validate(out)

    with open(RADLEX_FILE_NAME, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file, ["id", "class_id", "term", "synonyms", "definition", "obsolete"])
        radlex_terms = [
            clean_radlex_term(row) for row in reader if row["obsolete"] == "FALSE" and row["id"].startswith("RID")
        ]
    return radlex_terms


class RadLexFinding(BaseModel):
    """A RadLex concept which corresponds to a finding a rdaiologist might make in
    a radiology report. These could be either observations or diagnoses."""

    id: str = Field(description="The unique identifier for the RadLex concept.", pattern=r"^RID\d+$")
    term: str = Field(description="The canonical term for the RadLex concept.")
    type: Literal["observation", "diagnosis"] = Field(
        description="The type of the concept. Can be either 'observation' or 'diagnosis'."
    )
    reportable: bool = Field(
        description="""Whether the finding is a specific term that a radiologist would describe in a radiologist report. 
        For example, a radiologist wouldn't describe 'regurgitation' in a report (it's too general), 
        but they might describe 'mitral valve regurgitation'. 
        They wouldn't describe 'esophageal disorder', but they would describe 'achalasia.' 
        They wouldn't describe 'hernia', but they might describe 'inguinal hernia' or 'peristomal hernia'.
        If the finding is too general or not specific enough to be used in a report, this should be False.""",
    )


async def extract_findings(
    client: instructor.AsyncInstructor, terms: list[RadLexTerm], model: str = "gpt-4o-mini"
) -> list[RadLexFinding]:
    """Extract findings from a batch of RadLexTerms."""

    term_str = "\n".join(term.pretty_str(include_synonyms=True) for term in terms)

    # TODO: Make the prompt much better
    #   - Give it a system prompt that it's a medical informaticist helping to organize
    #     terms from an ontology
    #   - Let it know we want it to use the real terms, not improve them
    #   - Let it know NOT to include terms that are not findings, like anatomy
    #   - Don't include synonyms in the output

    # TODO: Make sure we're using appropriate retries
    # TODO: Make the interface nicer using TQDM
    finding_info = await client.chat.completions.create(
        model=model,
        response_model=list[RadLexFinding],
        messages=[
            {
                "role": "system",
                "content": "You are a medical informaticist tasked with extracting findings from an ontology. "
                "A finding is either an observation or a diagnosis that a radiologist would make in a report. "
                "Do not extract terms that are not findings, such as anatomical terms."
                "Do not include synonyms in the output."
                "Use only standard terms. Do not improve or modify the terms.",
            },
            {
                "role": "user",
                "content": f"Extract the findings from the following set of RadLex concepts:\n{term_str}",
            },
        ],
    )
    return finding_info


BATCH_SIZE = 16


async def main() -> None:
    load_dotenv()

    radlex_terms = load_radlex_terms()
    print(f"Loaded {len(radlex_terms)} RadLex terms.")

    # Initialize the OpenAI client
    client = instructor.from_openai(AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]))
    print("Starting extraction...")
    # TODO: Find an appropriate batch size that gives good results
    # TODO: See which models give the best results
    # TODO: Maybe we can use OpenRouter to try some non-OpenAI models (Claude, Gemini, DeepSeek?)
    offset = 13449
    findings = await extract_findings(client, radlex_terms[offset : offset + BATCH_SIZE])
    for finding in findings:
        print(finding)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
