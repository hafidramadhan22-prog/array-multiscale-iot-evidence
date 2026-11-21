# Array main manuscript and supplementary

Main document: `main.tex` (19-page PDF preview `main.pdf`). Companion: `SUPPLEMENTARY/supplement.tex` (4-page PDF preview).
The latest supervisor QUALITY_CONTROL_FINAL DOCX files dated 21 November are the prose authority. All seven main tables, five figures, five supplementary tables and four native equations are included. Availability paragraphs now link to the public R1–R4 repository and release qc-20261121. Numerical results were not changed.

Repository: https://github.com/hafidramadhan22-prog/array-multiscale-iot-evidence
Release: https://github.com/hafidramadhan22-prog/array-multiscale-iot-evidence/releases/tag/qc-20261121

Compile from this folder: `tectonic --untrusted main.tex`. Then enter SUPPLEMENTARY and run `tectonic --untrusted supplement.tex`. On Overleaf select main.tex as the main document; compile the supplementary separately. These are editable native LaTeX sources, not embedded manuscript PDF pages. Figure PDFs and their Python source/input tables are supplied.

The supervisor's latest supplementary prose is retained in full and occupies four pages at readable 10pt. It exceeds the earlier three-page target; no scientific evidence was silently removed to meet that older length. Confirm this updated length with the supervisor before submission.

The evidence repository excludes raw provider datasets and fitted estimator binaries. Its saved-results verifiers do not train, deserialize estimators or produce new classifier predictions. R1 preserves historical comparisons, including RF/SVM/KNN. Local raw and historical archives remain unchanged.

Packaging and verification scripts were prepared with ChatGPT/Codex assistance. This package is for author review, not a record of submission or final author approval.
