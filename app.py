import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

from models import GeneratedSuite, GeneratedTest
from policy_engine import vacation_days
from rag import generate_live_suite, load_demo_suite, retrieve_policy


ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / "policies" / "vacation.md"
DEMO_PATH = ROOT / "fixtures" / "demo_suite.json"


@dataclass(frozen=True)
class CaseResult:
    case: GeneratedTest
    actual_days: int
    passed: bool


def run_suite(suite: GeneratedSuite) -> list[CaseResult]:
    results = []
    for case in suite.cases:
        actual_days = vacation_days(case.months_employed)
        results.append(
            CaseResult(
                case=case,
                actual_days=actual_days,
                passed=actual_days == case.expected_days,
            )
        )
    return results


def print_report(evidence: list[Document], results: list[CaseResult]) -> None:
    print("RuleCheck")
    print("=========")
    print("\nPolicy evidence")
    for index, document in enumerate(evidence, start=1):
        text = " ".join(document.page_content.split())
        print(f"{index}. {text}")

    print("\nGenerated tests")
    print(f"{'Case':28} {'Months':>6} {'Expected':>8} {'Actual':>6} {'Result':>6}")
    print("-" * 62)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{result.case.name:28} "
            f"{result.case.months_employed:>6} "
            f"{result.case.expected_days:>8} "
            f"{result.actual_days:>6} "
            f"{status:>6}"
        )

    passed = sum(result.passed for result in results)
    print(f"\n{passed}/{len(results)} cases passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and run vacation-policy boundary tests."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="use the checked-in model response instead of calling OpenAI",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        load_dotenv(ROOT / ".env")
        evidence = retrieve_policy(POLICY_PATH)
        if args.demo:
            suite = load_demo_suite(DEMO_PATH)
        else:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set; add it to .env or run with --demo"
                )
            suite = generate_live_suite(
                evidence,
                model=os.getenv("OPENAI_MODEL", "gpt-5.6"),
            )
        results = run_suite(suite)
    except Exception as error:
        print(f"RuleCheck could not run: {error}", file=sys.stderr)
        return 2

    print_report(evidence, results)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
