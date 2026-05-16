from langchain_core.documents import Document

import json


def build_search_text(item: dict) -> str:

    return f"""
Assessment Name: {item.get("name", "")}

Description:
{item.get("description", "")}

Job Levels:
{", ".join(item.get("job_levels", []))}

Languages:
{", ".join(item.get("languages", []))}

Duration:
{item.get("duration", "")}

Assessment Type:
{", ".join(item.get("keys", []))}

Remote Testing:
{item.get("remote", "")}

Adaptive Testing:
{item.get("adaptive", "")}
""".strip()


def preprocess_documents(json_path: str):

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    documents = []

    for item in raw_data:

        text = build_search_text(item)

        metadata = {
            "entity_id": str(item.get("entity_id", "")),

            "name": item.get("name", ""),

            "url": item.get("link", ""),

            "job_levels": ", ".join(
                item.get("job_levels", [])
            ),

            "languages": ", ".join(
                item.get("languages", [])
            ),

            "duration": item.get("duration", ""),

            "keys": ", ".join(
                item.get("keys", [])
            ),

            "remote": str(
                item.get("remote", "")
            ),

            "adaptive": str(
                item.get("adaptive", "")
            )
        }

        doc = Document(
            page_content=text,
            metadata=metadata
        )

        documents.append(doc)

    return documents