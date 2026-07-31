from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from models import GeneratedSuite
from rag import (
    KeywordEmbeddings,
    build_generation_prompt,
    generate_live_suite,
    load_demo_suite,
    retrieve_policy,
)


ROOT = Path(__file__).resolve().parents[1]


def test_keyword_embeddings_are_deterministic_and_fixed_size():
    embeddings = KeywordEmbeddings()

    first = embeddings.embed_query("vacation after 12 months")
    second = embeddings.embed_query("vacation after 12 months")

    assert len(first) == 64
    assert first == second
    assert sum(value * value for value in first) == pytest.approx(1.0)


def test_retrieve_policy_returns_both_service_thresholds():
    evidence = retrieve_policy(ROOT / "policies" / "vacation.md")
    combined = " ".join(document.page_content for document in evidence)

    assert evidence
    assert "12" in combined
    assert "60" in combined


def test_retrieve_policy_reports_an_actionable_missing_file(tmp_path):
    missing_path = tmp_path / "vacation.md"

    with pytest.raises(
        FileNotFoundError,
        match=r"Policy file not found: .*vacation\.md; restore policies/vacation\.md",
    ):
        retrieve_policy(missing_path)


def test_load_demo_suite_validates_five_cases():
    suite = load_demo_suite(ROOT / "fixtures" / "demo_suite.json")

    assert len(suite.cases) == 5
    assert suite.cases[-1].months_employed == 60


def test_generation_prompt_contains_evidence_and_boundary_instruction():
    evidence = [
        Document(page_content="At 12 months employees receive 10 days."),
        Document(page_content="At 60 months employees receive 15 days."),
    ]

    prompt = build_generation_prompt(evidence)

    assert "At 12 months employees receive 10 days." in prompt
    assert "At 60 months employees receive 15 days." in prompt
    assert "exactly five" in prompt
    assert "immediately below and at each service threshold" in prompt


def test_generate_live_suite_uses_structured_output_boundary():
    suite = load_demo_suite(ROOT / "fixtures" / "demo_suite.json")
    calls = []

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_parsed=suite)

    fake_client = SimpleNamespace(responses=FakeResponses())
    evidence = [Document(page_content="Employees receive 10 days at 12 months.")]

    result = generate_live_suite(evidence, client=fake_client, model="test-model")

    assert result == suite
    assert calls == [
        {
            "model": "test-model",
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You generate candidate implementation tests from supplied "
                        "employee-policy evidence. Return only schema-valid cases."
                    ),
                },
                {
                    "role": "user",
                    "content": build_generation_prompt(evidence),
                },
            ],
            "text_format": GeneratedSuite,
        }
    ]


def test_generate_live_suite_rejects_empty_parsed_output():
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=lambda **kwargs: SimpleNamespace(output_parsed=None)
        )
    )

    with pytest.raises(ValueError, match="the model did not return a test suite"):
        generate_live_suite(
            [Document(page_content="Policy evidence.")],
            client=fake_client,
        )
