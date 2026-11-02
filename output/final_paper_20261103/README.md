# Final P8 manuscript

Authors, in the supplied order: **Yunjie Guo, Hanqing Yang**.

The English manuscript contains nine main sections, references, four appendices, eleven tables and two figures. The verified PDF has fourteen pages. No affiliation or target-venue format was supplied.

- `paper.pdf`: the final, visually checked PDF.
- `paper.md`: editable full text, generated tables and figure references.
- `paper.tex`: editable standalone LaTeX source; figures are in `figures/`. It uses standard `article`, `geometry`, `mathptmx`, `microtype`, `graphicx`, `longtable`, `booktabs`, `array`, `hyperref`, `caption` and `url` packages. Compile with `pdflatex paper.tex` twice. A LaTeX engine was unavailable in the local environment, so this source received static checks; the delivered PDF was generated independently with ReportLab and visually checked.
- `tables/`: manuscript tables and complete saved checkpoint/within-issue result tables.
- `MANUSCRIPT_MANIFEST.json`, `PAPER_QA.json`: source/result bindings, file hashes and quality checks.
- `authoring/`: the template and generator used in the original research project. Rebuilding from locked analyses requires that project's source tree and saved results; writing the manuscript does not refit models or create new predictions.

All new findings are exploratory. AI-assisted review, AI adjudication, earlier snapshot exposure, bootstrap refit failures and numerically saturated subgroup outputs are disclosed. A positive nonlinear secondary contrast does not replace the primary logistic result, whose interval includes zero. Some subgroup percentile intervals are not reliable inferential evidence.

This package is the 14-page PDF and its editable source. The repository root [`paper.pdf`](../../paper.pdf) is the later 13-page manuscript. The 27 October 2026 frozen-protocol paper is [`paper_20261027.pdf`](../../paper_20261027.pdf). Raw trajectories and fitted artifacts are not in this package.
