"""Record a candidate's files without creating a Git tag or claiming course approval."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/course_project/candidate_manifest.json"


def main():
    files = set()
    for name in (
        "README.md",
        ".gitignore",
        ".gitattributes",
        ".gitmodules",
        ".env.example",
        "AI_USAGE.md",
        "CONTRIBUTIONS.md",
        "LICENSES.md",
        "SECURITY.md",
        "scripts/prepare_course_evidence.py",
        "scripts/check_course_project.py",
        "scripts/replay_course_results.py",
        "scripts/render_course_tables.py",
        "scripts/build_course_documents.sh",
        "scripts/snapshot_course_candidate.py",
        "tests/test_course_evidence.py",
        "tests/test_course_replay.py",
    ):
        files.add(ROOT / name)
    for folder in (
        "docs/course_project",
        "docs/course_project_requirement",
        "report",
        "presentation",
        "results/course_project",
    ):
        files.update(
            p for p in (ROOT / folder).rglob("*") if p.is_file() and p != OUTPUT
        )
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }
    result = {
        "status": "submission_candidate_with_recorded_team_instructions",
        "date": "2026-11-08",
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "template_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT / "templates/UM_Typst_Template",
            text=True,
        ).strip(),
        "final_course_tag": None,
        "closeout_record": "docs/course_project/release_status.md",
        "files": hashes,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print("CANDIDATE_SNAPSHOT:", len(hashes), "files; no Git commit or tag created")


if __name__ == "__main__":
    main()
