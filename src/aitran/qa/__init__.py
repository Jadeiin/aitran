"""Quality-assurance module for translation units.

Runs translate-toolkit rule-based checks and returns structured results
for the LLM reviewer to consume as context.
"""

from aitran.qa.checkers import build_checker
from aitran.qa.runner import QAError, QARunner, Severity, UnitQAReport

__all__ = [
    "QAError",
    "QARunner",
    "Severity",
    "UnitQAReport",
    "build_checker",
]
