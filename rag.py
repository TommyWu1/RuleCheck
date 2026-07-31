import hashlib
import math
import re
from pathlib import Path
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from models import GeneratedSuite


RETRIEVAL_QUERY = (
    "vacation entitlement boundary tests for completed months of service "
    "at 12 and 60 months"
)


class KeywordEmbeddings(Embeddings):
    def __init__(self, size: int = 64) -> None:
        self.size = size

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.size
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def retrieve_policy(policy_path: Path, limit: int = 3) -> list[Document]:
    if not policy_path.is_file():
        raise FileNotFoundError(
            f"Policy file not found: {policy_path}; restore policies/vacation.md"
        )
    policy = policy_path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=180, chunk_overlap=20)
    chunks = splitter.split_documents([Document(page_content=policy)])
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=KeywordEmbeddings(),
        collection_name=f"rulecheck-{uuid4().hex}",
    )
    return vector_store.similarity_search(
        RETRIEVAL_QUERY,
        k=min(limit, len(chunks)),
    )


def load_demo_suite(path: Path) -> GeneratedSuite:
    return GeneratedSuite.model_validate_json(path.read_text(encoding="utf-8"))


def build_generation_prompt(evidence: list[Document]) -> str:
    numbered_evidence = "\n\n".join(
        f"[{index}] {document.page_content}"
        for index, document in enumerate(evidence, start=1)
    )
    return (
        "Generate exactly five boundary-focused test cases. Use only the policy "
        "evidence below. Do not invent rules. Include cases immediately below "
        "and at each service threshold.\n\n"
        f"Policy evidence:\n{numbered_evidence}"
    )


def generate_live_suite(
    evidence: list[Document],
    client: OpenAI | None = None,
    model: str = "gpt-5.6",
) -> GeneratedSuite:
    response = (client or OpenAI()).responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You generate candidate implementation tests from supplied "
                    "employee-policy evidence. Return only schema-valid cases."
                ),
            },
            {"role": "user", "content": build_generation_prompt(evidence)},
        ],
        text_format=GeneratedSuite,
    )
    if response.output_parsed is None:
        raise ValueError("the model did not return a test suite")
    return response.output_parsed
