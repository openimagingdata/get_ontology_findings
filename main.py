import os
from typing import Literal, Optional

import instructor
from dotenv import load_dotenv
from openai import OpenAI
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


def extract_findings(report: str) -> list[Finding]:
    """Extract findings from a radiology report."""

    client = instructor.from_openai(
        # OpenAI(
        #     base_url="http://localhost:11434/v1",
        #     api_key="Keys? We don't need no stinking keys!",
        # ),
        # mode=instructor.Mode.JSON,
        OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    )
    finding_info = client.chat.completions.create(
        model="gpt-4o-mini",
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
