# Sources, attribution and redistribution status

Checked 2026-11-08. This records observed provenance and unresolved permissions, not legal approval.

| Component | Source / identity | Observed license and action |
|---|---|---|
| DiffOSeg paper | Zhang et al., 2025, arXiv:2507.13087v2, https://arxiv.org/abs/2507.13087 | Cite the paper. The report paraphrases method details and cites numeric results |
| Imported DiffOSeg code | https://github.com/string-ellipses/DiffOSeg ; historical source identity in manifest.json | No root license was found in this checkout or visible upstream root listing. Permission for redistribution remains UNVERIFIED; do not invent an open-source license |
| P/A modifications | Exact frozen pa_source.tar.gz and separately hashed metrics dependency | Keep original notices; new coursework packaging does not relicense inherited code |
| Preprocessed LIDC PKL | Download link in INSTALL.md; SHA-256 6c401c96b0e1737d43fdccfc446cb0c3601473d3b7668ad4a68d22ec0c1ee81b | Specific processed-file license and transformation provenance require confirmation. Raw collection information: https://www.cancerimagingarchive.net/collection/lidc-idri/ ; that page could not be fetched during assembly. No medical data redistributed |
| D-Persona split/rank precedent | https://github.com/ycwu1997/D-Persona ; cited commit in docs/e11_lidc.md | Attribution retained. Audit each copied component's notice before external release |
| SAM2.1 optional E extension | facebookresearch/sam2 revision 2b90b9f5ceec907a1c18123530e92e794ad901a4 | Archived public runtime LICENSE says Apache-2.0. Official page: https://github.com/facebookresearch/sam2/blob/main/LICENSE . Public weight hash and runtime receipt remain in the experiment archive |
| Original SAM / other exploratory pretrained assets | Existing experiment-specific receipts and protocol documents | Not bundled. Model-specific source, license and redistribution terms must be reviewed if included in the selected submission |
| Python dependencies | pyproject.toml and uv.lock, plus archived lock for P/A | Package-specific licenses apply; dependency binaries are not bundled |

No blanket project LICENSE has been assigned by this assembly. Private local evidence preparation and public redistribution are separate decisions. Confirm the instructor's permitted submission route and any required medical-data approvals before representing the package as cleared.

## User-selected document template

UM_Typst_Template: https://github.com/HaroldHuanrongLIU/UM_Typst_Template , pinned submodule ac1964afca6a99bb60cb29ed2c2ddece00eb2eea. Original template design/source/components/documentation are copyright 2026 HUANRONG LIU under CC BY 4.0, with university identity assets and user content excluded. The original submodule is unchanged. The course slides adapt its main entry point and retain its footer; the A4 report reuses palette/logo with attribution. Typst compiler and Touying exporter licenses are supplied by their distributions.
