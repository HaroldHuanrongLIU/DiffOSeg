#!/usr/bin/env bash
# Build the report and both slide formats without using the training environment.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export UV_CACHE_DIR="$project_root/.course-build/uv-cache"
export UV_PROJECT_ENVIRONMENT="$project_root/.course-build/document-venv"
export TYPST_PACKAGE_CACHE_PATH="$project_root/.course-build/typst-packages"
mkdir -p "$TYPST_PACKAGE_CACHE_PATH"
python3 -B scripts/render_course_tables.py
uv sync --project presentation/toolchain --locked
uv run --project presentation/toolchain --locked --no-sync touying compile report/course_project.typ --root "$project_root" --format pdf --output report/course_project.pdf
uv run --project presentation/toolchain --locked --no-sync touying compile presentation/course_project.typ --root "$project_root" --format pdf --output presentation/course_project.pdf
uv run --project presentation/toolchain --locked --no-sync touying compile presentation/course_project.typ --root "$project_root" --format pptx --ppi 144 --output presentation/course_project.pptx
