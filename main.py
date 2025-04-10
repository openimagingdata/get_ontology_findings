import os
from typing import Literal, Optional

import instructor
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, Field

CXR_REPORT = """
 There is no focal consolidation, pleural effusion or pneumothorax.  Bilateral
 nodular opacities that most likely represent nipple shadows. The
 cardiomediastinal silhouette is normal.  Clips project over the left lung,
 potentially within the breast. The imaged upper abdomen is unremarkable.
 Chronic deformity of the posterior left sixth and seventh ribs are noted.
 """


class Finding(BaseModel):
    """A finding in a radiology report."""

    finding_name: str = Field(
        description="The name of the finding using standard language, even if different from the text in the report."
    )
    presence: Literal["present", "absent"] = Field(
        description="Whether the finding is described as present or absent in the report."
    )
    change_from_previous: Optional[Literal["unchanged", "improved", "worsened", "stable"]] = Field(
        description="Whether the finding has improved, worsened, or remained stable compared to a previous report."
    )
    attributes: Optional[dict[str, str | int | float]] = Field(
        description="Additional attributes of the finding, such as size or location."
    )


def openai_client() -> OpenAI:
    """Create an OpenAI client."""
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://api.openai.com/v1",
        organization=os.environ["OPENAI_ORGANIZATION"],
    )


def local_client() -> OpenAI:
    """Create a local OpenAI client."""
    return OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="vllm",
    )


def azure_openai_client() -> OpenAI:
    """Create an Azure OpenAI client."""
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

    client = AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint="https://autoprotocoling.openai.azure.com/",
        azure_ad_token_provider=token_provider,
    )
    return client


VLLM_MODEL = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
OPENAI_MODEL = "gpt-4o-mini"


def extract_findings(report: str) -> list[Finding]:
    """Extract findings from a radiology report."""

    client = instructor.from_openai(local_client())
    finding_info = client.chat.completions.create(
        model=VLLM_MODEL,
        response_model=list[Finding],
        messages=[
            {
                "role": "user",
                "content": f"Extract the findings from the following radiology report: {report}",
            }
        ],
    )
    return finding_info


def main() -> None:
    load_dotenv()

    print("Extracting findings from the CXR report...")
    finding_info = extract_findings(report=CXR_REPORT)

    for finding in finding_info:
        print(finding.model_dump_json(indent=2, exclude_none=True))
        print("---")


if __name__ == "__main__":
    main()
