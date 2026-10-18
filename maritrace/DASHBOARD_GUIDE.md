# MARITRACE dashboard

## Launch

From the repository root, install the existing project dependencies and start Streamlit:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard reads roots from `maritrace_config.yaml`, with optional `MARITRACE_DATA_ROOT` and `MARITRACE_OUTPUT_ROOT` environment-variable overrides. It then checks the local `runs\prod_outputs` mirror and `outputs` folder. Sample scenes under `data\sample` are generated demonstration data and are marked synthetic in the map view. The application does not run model training or expand the multi-gigabyte AIS/GFW sources during page rendering.

## Artifact expectations

Processed trajectories and drift layers are read from `outputs\processed_data`. Evaluation reports and trained models are read from `outputs\models`. Audit verification reads `outputs\audit\audit_log.json`. Missing files remain unavailable in the interface; run the corresponding existing pipeline stage to generate supported artifacts.

The current mirror contains model weights, model metrics, inventory, and synthetic samples. It does not currently contain processed AIS trajectories, processed drift outputs, an audit log, or the incident CSV. The ocean-current NetCDF discovered at the configured data path is zero bytes and is flagged as invalid.

## Scientific interpretation

The historical AIS snapshot dated January 2, 2024 is not event-aligned with NOAA incident 10716, which opened January 7. Its tracks must not be treated as event evidence. The incident opening date is not assumed to be discharge time. Drift fallback parameters are scenario assumptions, not observed environmental measurements. Classifier scores and vessel gear predictions are decision-support outputs and do not establish spill confirmation or responsibility.

The audit hash-chain verifier checks recorded chain integrity only. It does not establish the authenticity or truth of original evidence.
