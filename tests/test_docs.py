import re
from pathlib import Path

import pytest

CODE_BLOCK_RE = re.compile(r"```[ \t]*(python|py)[^\n]*\n(.*?)```", re.IGNORECASE | re.DOTALL)
DOCS_DIR = Path(__file__).parent.parent / "docs"


@pytest.mark.parametrize("md_file", sorted(DOCS_DIR.rglob("*.md")))
def test_python_code_blocks_in_docs(md_file: Path) -> None:
    """Test code block examples in docs."""
    assert md_file.suffix == ".md", f"Expected a Markdown file, got {md_file}"
    failures: list[str] = []
    text = md_file.read_text(encoding="utf-8")
    for match in CODE_BLOCK_RE.finditer(text):
        line_no = text.count("\n", 0, match.start()) + 1
        code = match.group(2)
        namespace = {"__name__": "__main__", "__file__": str(md_file)}
        try:
            exec(compile(code, f"{md_file}:{line_no}", "exec"), namespace)  # noqa: S102
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{md_file}:{line_no}\n{type(exc).__name__}: {exc}\nCode:\n{code}")

    assert not failures, f"{len(failures)} Python code block(s) failed:\n\n" + "\n\n".join(failures)
