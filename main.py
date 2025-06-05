import csv
import os
import random
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
    specificity: Literal["specific", "general"] = Field(
        description="The specificity of the concept, whether it refers to a specific finding or a general category of findings."
    )
    finding_or_attribute: Literal["finding", "attribute"] = Field(
        description="Whether the concept is something being described or a property of something being described."
    )


MAX_SIMULTANEOUS_REQUESTS = 10


async def extract_findings(
    client: instructor.AsyncInstructor, terms: list[RadLexTerm], model: str = "gpt-4o-mini"
) -> list[RadLexFinding]:
    """Extract findings from a batch of RadLexTerms."""

    # TODO: Make the prompt much better
    #   - Give it a system prompt that it's a medical informaticist helping to organize
    #     terms from an ontology
    #   - Let it know we want it to use the real terms, not improve them
    #   - Let it know NOT to include terms that are not findings, like anatomy
    #   - Don't include synonyms in the output

    # TODO: Make sure we're using appropriate retries
    # TODO: Make the interface nicer using TQDM

    SYSTEM_PROMPT = """
    You are a medical informaticist tasked with extracting findings from an ontology. 
    A finding is either an observation or a diagnosis that a radiologist would make in a report. 
    Do not extract terms that are not findings, such as anatomical structures or imaging techniques.
    For example, "l2 root of femoral nerve", "external granular layer of left Brodmann area 2", "ureteral
    proper wall", "cochlear septum", and "premotor cortex" are anatomical structures and should not
    be included in the output.
    Do not include synonyms in the output.
    Use only standard terms. Do not improve or modify the terms.
    """

    def make_user_prompt(term_str: str) -> str:
        return f"""
        Include whether the finding is a specific term that a radiologist would describe in a radiologist report
        or a general category. 
        For example, a radiologist wouldn't describe 'regurgitation' in a report (it's too general), 
        but they might describe 'mitral valve regurgitation'. 
        They wouldn't describe 'esophageal disorder', but they would describe 'achalasia.' 
        They wouldn't describe 'hernia', but they might describe 'inguinal hernia' or 'peristomal hernia'.

        Also include whether the concept refers to a finding or an attribute of a finding. For example,
        "vascular calcification" is a finding, while "calcified" is an attribute of a finding. "Liver lesion" is a finding,
        while "T2 hyperintense" is an attribute of a finding.

        Extract the findings from the following set of RadLex concepts:
        {term_str}
        """

    results: list[RadLexFinding] = []
    for offset in range(0, len(terms), 4):
        term_set = terms[offset : offset + 4]
        term_str = "\n".join(term.pretty_str(include_synonyms=True) for term in term_set)
        finding_info = await client.chat.completions.create(
            model=model,
            response_model=list[RadLexFinding],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": make_user_prompt(term_str),
                },
            ],
        )
        if finding_info:
            results.extend(finding_info)
    return results


BATCH_SIZE = 32


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
    random_batch = random.sample(radlex_terms, BATCH_SIZE)
    print(f"Extracting findings from {len(random_batch)} random RadLex terms.")
    findings = await extract_findings(client, random_batch)
    for finding in findings:
        print(finding)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
