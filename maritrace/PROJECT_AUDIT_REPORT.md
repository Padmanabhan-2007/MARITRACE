# MARITRACE Project Audit

**Audit date:** 2026-10-18  
**Scope:** Dashboard launch and data handling, pipeline evidence flow, decision support, dependency setup, and front-end presentation.

## Issues found and fixes applied

| Priority | Issue | Fix |
|---|---|---|
| Critical | The pipeline assigned a fixed `0.942` SAR score to every incident and described it as calibrated incident confidence. It was based on a saved evaluation result, not a scene linked to the incident. | The pipeline now passes no score when there is no incident-specific SAR scene. The fusion engine records `ABSTAIN / MISSING_INCIDENT_SAR` and does not rank vessels. |
| Critical | A free-text acquisition note containing words such as “confirm” or “positive” could automatically change a lead to `CONFIRMED_ENFORCEMENT_TARGET` and alter scores. | Text notes now require independent review. They cannot confirm or exculpate a vessel or change existing lead scores. |
| High | Dependency declarations omitted packages imported by core paths (`torchvision`, PyYAML, zstandard, NetCDF support, and seaborn); the original vision pins did not have a compatible wheel for the machine's Python 3.13 runtime. | Declared the missing packages, matched torchvision 0.21.0 to torch 2.6.0, and moved scikit-image to 0.26.0 for available Python 3.13 wheels. Kept the same stack compatible with the Python 3.11 Docker base. |
| High | The dashboard hard-coded an external Windows dataset location, making artifact discovery checkout-specific. | Data and output roots now come from `maritrace_config.yaml`, with `MARITRACE_DATA_ROOT` and `MARITRACE_OUTPUT_ROOT` overrides and local mirror fallbacks. |
| High | Existing zero-byte files were reported as “Available,” and the sidebar could mark empty model files ready. | Artifact inventory now reports zero-byte files as invalid; sidebar readiness requires a non-empty file. |
| Medium | The original overview looked like a generic Streamlit template and did not distinguish a demo from a real incident workspace. | Rebuilt the interface as a dark maritime operations desk with case provenance, a status-led layout, six focused workflows, and visible synthetic-data labels. |
| Medium | Generated briefings could state that a missing drift input was a corrupted zero-byte NetCDF, and the pipeline omitted actual drift provenance when it was available. | Briefings now distinguish missing provenance from a confirmed zero-byte file, and the pipeline passes drift metadata into the briefing generator. |

## Remaining data and validation gaps

- The local mirror has model artifacts and synthetic scenes, but it lacks the processed AIS trajectories, drift outputs, audit log, NOAA incident CSV, and a SAR scene linked to the selected incident.
- The configured ocean-current NetCDF is reported as zero bytes when present. Fallback drift values remain scenario assumptions.
- The available January 2, 2024 AIS snapshot predates incident 10716’s January 7 opening. It is a regional baseline, not incident evidence.
- Evaluation metrics describe model test sets; they do not establish performance on an incident scene or prove vessel responsibility.
- The dashboard basemap uses OpenStreetMap tiles and requires network access to render the basemap.
- The shared Python 3.13 interpreter is not the project runtime. Use the repository `.venv`, which now installs the compatible model stack and pins, to run the dashboard and pipeline.
- The earlier one-off Streamlit dependency installation changed the shared global Pillow package to 11.3.0, which conflicts with an unrelated installed `pdfplumber` package. The project pins Pillow 11.0.0; use the documented virtual environment to keep project dependencies isolated.

## Verification status

The dashboard and local API are running from the project `.venv`. Python compilation, a missing-SAR abstention smoke check, model load/inference checks, and the focused dashboard/core suite passed. Full incident attribution remains blocked by the missing case-linked sources described below; no live incident conclusion is produced.

## Product workflow and model runtime review

- Reorganized the dashboard navigation into **Command Center**, **Investigate**, **Model Lab**, and **Records & Provenance**. The Command Center now provides a deterministic synthetic analysis action that calls the repository's scenario, suspect-ranking, and drift functions and clearly labels every result as synthetic.
- Added optional browser speech synthesis and speech dictation controls for an investigator's session note. The microphone is only requested after the user selects dictation. Browser speech recognition may use the browser vendor's service; the transcript is displayed for review and manual copy into the session note, and never changes evidence validation or scores.
- Initially, the shared Python runtime could not load the saved models. The project now pins PyTorch 2.6.0 / torchvision 0.21.0 / scikit-learn 1.9.0 and scikit-image 0.26.0, and installs them in `.venv`. Both saved artifacts load there; the SAR checkpoint produced a smoke-test score and the vessel model returned class probabilities.
- Exercised FastAPI's `/health` and `/analyze` handlers directly: health returned `ok`, and the synthetic analyze handler returned 20 ranked candidates plus a 4-point drift path. The dashboard synthetic action also rendered its ranked output in the browser.
- The initial shared-runtime run passed 10 focused dashboard/core checks with three ML-dependent checks excluded. Follow-up testing in the project-local runtime passed all 13 checks; real incident inference still needs event-linked evidence.

## Follow-up: project-local ML environment

The machine has Python 3.13 but no Python 3.11 installation. The project originally pinned torch 2.5.1 / torchvision 0.20.1, and scikit-image 0.24.0 attempted a source build that failed without a C compiler. The pins now use torch 2.6.0 / torchvision 0.21.0 and scikit-image 0.26.0, which have Windows Python 3.13 wheels; the project `.venv` was created and all requirements installed, including seaborn for the vessel classifier module.

**Runtime verification:** in `.venv`, the saved SAR model loaded and returned a smoke-test prediction, the saved scikit-learn 1.9 model loaded and predicted gear classes, and the CSV feature-to-prediction path classified 258 synthetic AIS rows. The full test suite passes all 23 tests when its temp directory is placed in the writable workspace; the default Windows sandbox Temp path caused one initial permission failure. The running app was relaunched from `.venv` at `http://127.0.0.1:8502/`; its operations screen reports 2/2 runnable models and the local FastAPI demo online. `pip check` reports no broken requirements.

Real incident attribution remains unavailable because the checkout still lacks case-linked SAR, incident-time AIS tracks, validated currents, and processed audit/drift evidence. The tested API and synthetic scenario demonstrate software behavior only; they are not real maritime findings.

## Follow-up: full interface rebuild

The previous UI changes still looked like the original Streamlit template. The entry point now renders a new investigator desk in `src/dashboard_v2.py`: a dark marine palette, case slate with provenance, live repository/model/API readiness, and six workspaces for operations, dossier notes, tracks/drift, model checks, evidence inventory, and repeatable scenario simulation. The Streamlit theme configuration now matches the interface. Map tiles use public OpenStreetMap tiles instead of the Carto dark tile style that displayed a key-required watermark in this environment.

The operations screen explicitly shows the configured location as unvalidated, the case-linked source count as zero, both models as runnable in the project runtime, and the local demo API's live health status. Session notes are browser-session-only; optional dictation requests microphone access only after selection and does not affect scoring. Model Bench supports SAR image upload and AIS CSV classification, with the resulting model scores labeled as uncalibrated and not evidence of responsibility.

**Follow-up verification:** Python compilation passed for the new entry point and dashboard module. The focused dashboard/core suite passed (13 passed). The simulation helper returned 20 ranked synthetic candidates, 258 AIS reports, and four configured drift steps. The local FastAPI service is running on `127.0.0.1:8000`; `/health` returned `ok` and `/analyze?scene_id=scene-1` returned 20 synthetic candidates and four drift points. Browser inspection confirmed the rebuilt operations deck, 14/29 available repository artifacts, 2/2 runnable models, and the API online indicator. Real event analysis remains blocked by missing incident-linked AIS/SAR/current data; the dashboard/API demo is not a substitute for those inputs.
