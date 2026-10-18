# MARITRACE: Maritime Pollution Forensics & Evidence Fusion System

**Developed for the 24-Hour IBM Z / LinuxONE Datathon**  
*Autonomous AI pipeline fusing satellite Synthetic Aperture Radar (SAR), AIS vessel kinematics, Lagrangian ocean drift, and evidence-sufficiency decision support.*

---

## Architecture Overview

```mermaid
flowchart TD
    A[CSIRO Sentinel-1 SAR Dataset<br/>3,695 Class_0 / 1,843 Class_1] -->|Transfer Learning| B[MobileNetV2 SAR Classifier<br/>Acc: 94.8% | ROC-AUC: 0.986]
    C[NOAA Compressed AIS Stream<br/>ais-2024-01-02.csv.zst] -->|Stream Decompression| D[Kinematic Trajectory Reconstructor<br/>298 Vessels in Mississippi Delta]
    E[GFW Vessel Activity CSVs<br/>28.5M Rows / 6 Gear Classes] -->|GroupShuffleSplit by MMSI| F[RandomForest Gear Classifier<br/>Acc: 60.4% | Macro-F1: 0.480]
    G[Copernicus Current & Wind<br/>0-Byte Corrupt NC Detected] -->|Fallback Physics Engine| H[Lagrangian Backward Drift<br/>Expanding Origin Envelopes]
    
    B -->|P_oil Confidence| I[MARITRACE Evidence-Fusion Engine<br/>Multi-Factor Scoring & Explicit Abstention]
    D -->|Spatial-Temporal Overlap| I
    F -->|Vessel Context Risk| I
    H -->|Origin Cone Intersection| I
    
    I -->|Actionable Lead / Ambiguity / Abstain| J[Next-Best-Evidence Engine<br/>EVOI Policy-Based Acquisition Ranking]
    I -->|SHA-256 Hash Chaining| K[Cryptographic Tamper-Evident Audit Log]
    J -->|Structured Incident Dossier| L[Forensic LLM Advisory Layer]
```

---

## 1. Quick Start & Execution Guide (Windows PowerShell)

All commands are tested and executed locally in Windows PowerShell.

### Environment Setup
```powershell
# Navigate to workspace
cd "C:\Naresh things\maritrace"

# PyTorch 2.6 / torchvision 0.21 supports Python 3.11 and Python 3.13.
# Create and activate a project-local environment before installing packages.
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies (including zstandard, PyTorch CPU, scikit-learn, Streamlit)
python -m pip install -r requirements.txt
```

### Complete Pipeline Execution
```powershell
# Run the complete end-to-end pipeline (restartable; skips existing checkpoints)
python maritrace_pipeline.py --stage all

# Force retraining and re-extraction from scratch
python maritrace_pipeline.py --stage all --force
```

### Individual Pipeline Stages
```powershell
# 1. Recursive dataset discovery & integrity audit
python maritrace_pipeline.py --stage inventory

# 2. Train and evaluate MobileNetV2 satellite SAR classifier
python maritrace_pipeline.py --stage satellite

# 3. Train and compare vessel gear classifiers (Random Forest vs HistGradientBoosting)
python maritrace_pipeline.py --stage vessel

# 4. Stream decompress and reconstruct AIS trajectories in the Gulf of Mexico
python maritrace_pipeline.py --stage ais

# 5. Validate NetCDF and simulate Lagrangian back-drift origin envelopes
python maritrace_pipeline.py --stage drift

# 6. Run multi-source evidence fusion and next-best-evidence recommendations
python maritrace_pipeline.py --stage fusion
```

### Run Unit & Integration Tests
```powershell
# Execute complete 16-test suite
pytest -q
```

### Launch Interactive Forensics Dashboard
```powershell
python -m streamlit run app.py
```

The dashboard reads dataset and output folders from `maritrace_config.yaml`. Set
`MARITRACE_DATA_ROOT` or `MARITRACE_OUTPUT_ROOT` to override those folders for a
local checkout without editing the configuration file.

---

## 2. Model Performance & Empirical Benchmarks

### A. Satellite SAR Oil Slick Classifier (MobileNetV2 Transfer Learning)
- **Dataset:** CSIRO Sentinel-1 SAR Dataset (Australia / SE Asia).
- **Verified Classes:** `Class_0` = Clean ocean / look-alikes (3,695 chips); `Class_1` = Confirmed oil slicks (1,843 chips).
- **Splits:** 70% Train (3,876), 15% Validation (831), 15% Test (831) — Stratified & Deduplicated.
- **Hardware:** CPU Training (22 logical cores via multi-threaded PyTorch).

| Metric | Score | Detail |
|---|---|---|
| **Test Accuracy** | **94.83%** | 788 correct predictions out of 831 test images |
| **Precision (Oil)** | **89.80%** | Minimizes false accusations against legitimate traffic |
| **Recall (Oil)** | **95.31%** | Detects 264 of 277 verified mineral oil slicks |
| **F1-Score (Oil)** | **0.9247** | Balanced harmonic mean on imbalanced dataset |
| **Macro F1** | **0.9426** | Strong generalization across both classes |
| **ROC-AUC** | **0.9857** | Near-perfect class separation across decision thresholds |
| **Confusion Matrix** | `TN=524, FP=30, FN=13, TP=264` | Test set evaluation on completely unseen images |

*Model Artifacts:* `outputs/models/satellite_classifier/best_model.pth`, `plots/satellite_confusion_matrix.png`, `plots/satellite_roc_curve.png`.

---

### B. Vessel Activity & Gear Classification (Global Fishing Watch Benchmark)
- **Dataset:** Global Fishing Watch multi-gigabyte benchmark (`drifting_longlines`, `fixed_gear`, `pole_and_line`, `purse_seines`, `trawlers`, `trollers`).
- **Feature Engineering:** Kinematic transformations (`speed`, `course_sin`, `course_cos`, `speed_sq`), spatial logistics (`log_dist_shore`, `log_dist_port`), and temporal signatures.
- **Data Leakage Prevention:** Evaluated using **`GroupShuffleSplit` on vessel MMSI** — ensuring models are tested on 100% unseen ships.

| Model Architecture | Test Accuracy | Macro-F1 | Training Time |
|---|---|---|---|
| **HistGradientBoosting** | 58.11% | 0.4636 | 3.01s |
| **RandomForestClassifier (Selected)** | **60.40%** | **0.4801** | **0.69s** |

*Model Artifacts:* `outputs/models/vessel_classifier/vessel_gear_model.joblib`, `plots/vessel_confusion_matrix.png`, `plots/vessel_feature_importances.png`.

---

## 3. AIS Spatiotemporal Analysis & NOAA Incident Alignment

- **Compressed Stream:** `ais-2024-01-02.csv.zst` (189.13 MB compressed, ~4.77 million observations).
- **Incident Grounding:** NOAA Incident 10716 (*Mystery Sheen; vicinity of Coquille Bay, LA* at 29.1350° N, -89.3039° W).
- **Streaming Pipeline:** Streamed with `zstandard` in Python chunks without decompressing gigabytes to disk.
- **Reconstruction:** Filtered 46,237 observations in 2.09 seconds; reconstructed complete trajectories for **298 distinct vessels** in the Mississippi River Delta.
- **Closest trajectories in the January 2 regional historical baseline (not incident leads; NOAA incident opened January 7):**
  1. `MISS GWEN` (MMSI: 367747070) — Closest Approach: 4.05 km | AIS Gap: 0.15h | Speed: 13.3 kts
  2. `CAPT PHILLIP BOX` (MMSI: 367598980) — Closest Approach: 4.12 km | AIS Gap: 0.06h | Speed: 4.0 kts
  3. `ORIENT FOCUS` (MMSI: 255992000) — Closest Approach: 4.15 km | AIS Gap: 0.23h | Speed: 10.4 kts

---

## 4. Ocean Hydrodynamics & Drift Forensics

- **NetCDF Validation:** `cmems_obs-mob_glo_phy-cur_nrt_0.25deg_P1D-m_1792237857475.nc` was detected as **0 bytes (corrupted)**. The system reported the fault and safely applied calibrated regional hydrodynamic parameters (Copernicus Gulf of Mexico mean current $u=+0.18\,\text{m/s}, v=-0.08\,\text{m/s}$ and wind $u=-3.5\,\text{m/s}, v=-2.0\,\text{m/s}$).
- **Lagrangian Physics:** Implements the 3% wind factor with a 12° Coriolis deflection (Ekman spiral) and Monte Carlo turbulent eddy diffusion ($K_h = 15\,\text{m}^2/\text{s}$).
- **Outputs:** Expanding origin envelopes at -2h, -4h, -6h, -12h, -18h, -24h exported to GeoJSON and CSV.

---

## 5. Evidence-Fusion & Next-Best-Evidence Engines

### Strict Forensic Principles & Abstention
1. **Presumption of Non-Guilt:** Proximity alone never establishes legal guilt.
2. **Terrestrial AIS Gaps:** Antenna propagation dropouts must not be presumed malicious without satellite-AIS verification.
3. **Mandatory Abstention:**
   - Abstains when SAR confidence is low (look-alikes).
   - Abstains when no vessel entered the origin cone.
   - Emits `AMBIGUOUS_LEADS` when multiple vessels share indistinguishable scores.
   - Rejects time-mismatched AIS windows before ranking: historical baseline tracks appear only in `historical_baseline_vessels`, never `ranked_leads`.

**Uncertainty is a heuristic 0–1 decision-support index, not a calibrated Bayesian probability.** For example, a heuristic decrement from 0.65 to 0.30 is 0.35 absolute and 53.8% relative. Satellite scene-level generalization is not established by the available filename-range spatial-block manifest.

### Next-Best-Evidence Engine
Models **Expected Value of Information (EVOI)** across 5 acquisition modalities:
- Commercial Satellite-AIS Gap-Fill (Spire / exactEarth) — *Priority: 0.7563, Cost: $500, Latency: 2h*
- High-Resolution Optical Satellite (Sentinel-2 / PlanetScope) — *Priority: 0.6552, Cost: $250, Latency: 12h*
- Maritime Patrol Aircraft Overflight (USCG / SLAR) — *Priority: 0.5412, Cost: $4500, Latency: 4h*
- Port State Control Fuel & Bilge Sampling — *Priority: 0.4900, Cost: $1200, Latency: 36h*
- High-Resolution Copernicus Reanalysis — *Priority: 0.4678, Free, Latency: 8h*

---

## 6. Cryptographic Provenance & Audit Trail

- **Standard:** Built in alignment with ISO 27037 digital evidence admissibility guidelines.
- **Mechanism:** Every action (dataset hash, model checkpoint, drift simulation, fusion scoring, and user decision) is linked in a **cryptographic SHA-256 hash chain**.
- **Tamper Evidence:** Running `verify_integrity()` proves no record has been altered post-hoc.
- **Verification Command:**
  ```python
  from src.audit_logger import AuditTrail
  audit = AuditTrail()
  audit.load(r"D:\dataset maritrace\outputs\audit\audit_log.json")
  valid, err = audit.verify_integrity()
  print(f"Audit chain intact: {valid}")
  ```

---

## 7. Limitations & Scientific Disclosures

See [`LIMITATIONS_REPORT.md`](LIMITATIONS_REPORT.md) for full scientific disclosures regarding:
- 0-byte NetCDF file fallback.
- CSIRO SAR chip geographic bounds vs NOAA Gulf of Mexico AIS.
- VHF propagation limits causing innocent terrestrial AIS dropouts.
- Limits of cryptographic hash chains (proves log immutability, not external sensor veracity).
