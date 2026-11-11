# Slide assets and provenance

Updated 2026-11-11. The English 15-page deck uses the existing UM template.

## Published paper figures

Han Zhang, Xiangde Luo, Yong Chen and Kang Li (2025), *DiffOSeg: Omni Medical Image Segmentation via Multi-Expert Collaboration Diffusion Model*, arXiv:2507.13087v2. These figures retain the authors' original terms; the arXiv page lists the arXiv perpetual non-exclusive license. They are not relicensed under the template's CC BY 4.0.

Both downloaded PNGs remain byte-for-byte unchanged. Cropping is performed only by viewports in the slide source; it does not alter or synthesize scientific content.

- `diffoseg-paper-consensus.png`: [Figure 2; slide 2 displays the first CT/annotation row through a viewport.](https://arxiv.org/html/2507.13087v2/3_1.png) SHA-256 `91d550a794441b561700a12192a99fb403e1777c1a1196d6b3705d3d350a1c34`.
- `diffoseg-paper-framework.png`: [Figure 1; slide 3 clips unused outer whitespace through a viewport.](https://arxiv.org/html/2507.13087v2/main_workflow.png) SHA-256 `be47b49c77d3b6cd0e2e5dc4263edd4e26e0d49d61c3189d4af882939b9aeac9`.

## Evidence charts

`pa-paired-intervals.svg` reads `pa_results.json`; `preference-results.svg` reads `formal_e.json`. Generated with the existing pinned project matplotlib, not estimated from screenshots. P/A bars show paired 95% intervals; preference dots show point estimates only. The preference x axis is explicitly truncated and marked. The displayed baseline name is Reproduced DiffOSeg (wrapped across two lines); its frozen evidence key remains `e11`.

Regenerate from the repository root:

```bash
.venv/bin/python -B presentation/render_figures.py
```

Then export both formats with the existing pinned Touying toolchain, as used in `scripts/build_course_documents.sh`. The checked-in SVGs allow the document-only environment to compile without importing the training environment.

Full sources, interpretation limits and speaking notes: `docs/course_project/defense.md`.
