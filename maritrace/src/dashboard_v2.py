"""MARITRACE's investigator-focused maritime operations desk.

The interface distinguishes repository evidence, workflow estimates, and synthetic
demonstrations. A successful model file read is not treated as proof of inference.
"""
from __future__ import annotations

import html
import importlib.util
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import folium
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit.components.v1 import html as component_html
from streamlit_folium import st_folium

from .dashboard_data import ROOT, data_file, inventory, load_json, output_file
from .dashboard_maps import valid_coordinate_rows
from .dashboard_models import load_satellite_model, predict_satellite_score
from .drift import drift_path
from .score import rank_suspects
from .simulate import scenario

_C = {
    "bg": "#07141b", "panel": "#10232c", "panel2": "#132a34", "line": "#28434b",
    "text": "#e6f0ec", "muted": "#9bb0ae", "soft": "#6e8989", "aqua": "#54e0cf",
    "mint": "#c7f36b", "amber": "#ffbd66", "red": "#ff7b73", "blue": "#7aaeff",
}


def _load_incident() -> dict:
    cfg = {}
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / "maritrace_config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    inc = cfg.get("incident", {})
    return {
        "id": str(inc.get("default_incident_id", "10716")),
        "name": str(inc.get("name", "Mystery Sheen · Coquille Bay")),
        "lat": float(inc.get("lat", 29.135)), "lon": float(inc.get("lon", -89.3039)),
        "opened": str(inc.get("open_date", "2024-01-07")),
        "observed": str(inc.get("observation_time_utc", "unknown")),
    }


def _records_available() -> tuple[int, int]:
    rows = inventory()
    ready = sum(r["Status"] == "Available" for r in rows)
    return ready, len(rows)


def _api_status() -> tuple[bool, str]:
    """Check the local FastAPI demo endpoint without failing the dashboard."""
    try:
        with urlopen("http://127.0.0.1:8000/health", timeout=0.4) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if response.status == 200 and payload.get("ok") and payload.get("data") == "simulated":
            return True, "Synthetic FastAPI endpoint responding"
    except (URLError, TimeoutError, OSError, ValueError):
        pass
    return False, "Local FastAPI service is not responding"


def _models() -> list[dict]:
    torch_ok = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None
    skl_version = None
    try:
        import sklearn
        skl_version = sklearn.__version__
    except Exception:
        pass
    satellite = output_file("models/satellite_classifier/best_model.pth")
    vessel = output_file("models/vessel_classifier/vessel_gear_model.joblib")
    vessel_ready = False
    vessel_note = "Model artifact missing"
    if vessel:
        # Pickle/joblib payloads are version-sensitive. This repository pins its
        # vessel artifact's environment in requirements.txt, so do not advertise
        # inference readiness when this interpreter differs from that pin.
        pinned = None
        try:
            for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("scikit-learn=="):
                    pinned = line.split("==", 1)[1].strip()
                    break
        except OSError:
            pass
        if pinned and pinned != (skl_version or ""):
            vessel_note = f"Pinned for scikit-learn {pinned}; current runtime is {skl_version or 'unavailable'}"
        else:
            try:
                import joblib
                joblib.load(vessel)
                vessel_ready, vessel_note = True, "Artifact loads in this runtime"
            except Exception as exc:
                vessel_note = f"Cannot load artifact: {type(exc).__name__}"
    return [
        {"name": "SAR slick classifier", "file": satellite, "ready": bool(satellite and torch_ok), "note": "PyTorch and torchvision available" if torch_ok else "Inference blocked · PyTorch / torchvision not installed", "kind": "vision"},
        {"name": "Vessel gear classifier", "file": vessel, "ready": vessel_ready, "note": vessel_note, "kind": "tabular"},
    ]


def _asset_card(label: str, value: str, note: str, tone: str = "aqua") -> str:
    return f'<div class="asset-card"><div class="asset-label">{label}</div><div class="asset-value" style="color:{_C[tone]}">{value}</div><div class="asset-note">{note}</div></div>'


def _card(title: str, kicker: str, body: str, tone: str = "aqua") -> None:
    st.markdown(
        f'<div class="surface"><div class="kicker" style="color:{_C[tone]}">{kicker}</div>'
        f'<div class="surface-title">{title}</div><div class="surface-body">{body}</div></div>',
        unsafe_allow_html=True,
    )


def _demo_analysis() -> dict:
    s = scenario(42)
    ranked = rank_suspects(s["features"])
    path = drift_path(s["lat"], s["lon"], s["wind_u"], s["wind_v"], s["cur_u"], s["cur_v"])
    return {"scenario": s, "ranked": ranked, "path": path}


def _speech_tools() -> None:
    component_html("""
    <div style="font:13px Inter,system-ui,sans-serif;color:#c6d8d5;background:#10232c;border:1px solid #28434b;border-radius:12px;padding:13px 15px">
      <div style="font-weight:700;margin-bottom:7px;color:#54e0cf">VOICE TOOLS · OPTIONAL</div>
      <div style="display:flex;gap:9px;flex-wrap:wrap">
        <button onclick="speakText()" style="background:#54e0cf;color:#07141b;border:0;border-radius:7px;padding:8px 11px;font-weight:700;cursor:pointer">▶ Read note aloud</button>
        <button onclick="listen()" style="background:#132a34;color:#e6f0ec;border:1px solid #45636a;border-radius:7px;padding:8px 11px;font-weight:700;cursor:pointer">● Dictate a note</button>
        <button onclick="speechSynthesis.cancel()" style="background:#132a34;color:#e6f0ec;border:1px solid #45636a;border-radius:7px;padding:8px 11px;cursor:pointer">Stop</button>
      </div>
      <div id="status" style="margin-top:8px;color:#9bb0ae">Dictation asks for microphone access only when selected. Review and copy the transcript into the note field.</div>
      <div id="transcript" style="margin-top:8px;line-height:1.5"></div>
    </div>
    <script>
      function speakText(){let text=parent.document.querySelector('textarea')?.value||'';if(!text){document.getElementById('status').textContent='Write a note first.';return;}speechSynthesis.cancel();speechSynthesis.speak(new SpeechSynthesisUtterance(text));}
      function listen(){const R=window.SpeechRecognition||window.webkitSpeechRecognition;if(!R){document.getElementById('status').textContent='Speech recognition is not supported in this browser.';return;}let r=new R();r.lang='en-US';r.interimResults=true;document.getElementById('status').textContent='Listening…';r.onresult=e=>document.getElementById('transcript').textContent=Array.from(e.results).map(x=>x[0].transcript).join(' ');r.onerror=e=>document.getElementById('status').textContent='Dictation error: '+e.error;r.onend=()=>document.getElementById('status').textContent='Dictation ended. Review and copy the transcript if needed.';r.start();}
    </script>
    """, height=132)


def _map(lat: float, lon: float, demo: dict | None = None) -> None:
    # Public OSM tiles work without an API key; prior Carto dark tiles displayed
    # a key-required watermark in environments without a Carto credential.
    center = [demo["scenario"]["lat"], demo["scenario"]["lon"]] if demo else [lat, lon]
    m = folium.Map(location=center, zoom_start=7, tiles="OpenStreetMap", control_scale=True)
    if demo:
        s = demo["scenario"]
        folium.CircleMarker([s["lat"], s["lon"]], radius=7, color="#54e0cf", fill=True, fill_color="#54e0cf", fill_opacity=.9, tooltip="Synthetic spill anchor").add_to(m)
        folium.PolyLine([[p["lat"], p["lon"]] for p in demo["path"]], color="#54e0cf", weight=3, tooltip="Synthetic constant-flow drift path").add_to(m)
        positions = s["ais"].sort_values("timestamp").groupby("mmsi").tail(1)
        scores = demo["ranked"].set_index("mmsi")["score"].to_dict()
        for _, row in positions.iterrows():
            folium.CircleMarker([row["lat"], row["lon"]], radius=4, color="#7aaeff", fill=True, fill_color="#7aaeff", fill_opacity=.8, tooltip=f"Synthetic MMSI {row['mmsi']} · heuristic {scores.get(row['mmsi'], 0):.2f}").add_to(m)
    else:
        folium.CircleMarker([lat, lon], radius=8, color="#ffbd66", fill=True, fill_color="#ffbd66", fill_opacity=.9, tooltip="Configured incident location · verify against case evidence").add_to(m)
        folium.Marker([lat, lon], tooltip="Incident reference point").add_to(m)
    st_folium(m, height=485, use_container_width=True, returned_objects=[])


def _demo_panel() -> None:
    st.markdown('<div class="section-head"><span>SIMULATION LAB</span><span class="signal-tag">SYNTHETIC DATA · NEVER CASE EVIDENCE</span></div>', unsafe_allow_html=True)
    st.markdown("### Run a reproducible scenario")
    st.caption("Runs the included deterministic AIS scenario, explainable suspect ranking, and constant-flow drift approximation.")
    if st.button("▶  Run synthetic analysis", type="primary", key="run_demo"):
        st.session_state["demo_analysis"] = _demo_analysis()
    result = st.session_state.get("demo_analysis")
    if not result:
        _card("One click to explore the full pipeline", "REPRODUCIBLE · SEED 42", "Generate 20 synthetic candidates and a four-step drift estimate. Results are always visibly labeled synthetic.")
        return
    s, ranked = result["scenario"], result["ranked"]
    x1, x2, x3 = st.columns(3)
    x1.metric("Synthetic tracks", ranked.shape[0])
    x2.metric("Position reports", len(s["ais"]))
    x3.metric("Drift steps", len(result["path"]))
    st.markdown('<div class="signal-tag">DEMO RANKING — NOT A REAL-WORLD LEAD</div>', unsafe_allow_html=True)
    st.dataframe(ranked[["mmsi", "name", "ship_type", "minimum_distance_km", "loiter_fraction", "max_gap_hours", "score"]].head(20), use_container_width=True, hide_index=True)
    st.caption("Heuristic score combines proximity, low-speed loitering, AIS gap, and vessel type. It is not a probability of responsibility.")
    left, right = st.columns([1.05, 1])
    with left:
        _map(_load_incident()["lat"], _load_incident()["lon"], result)
    with right:
        _card("Drift path", "SIMPLIFIED PHYSICS", "Wind and currents are held constant. This is a demonstration estimate, not incident-specific oceanographic evidence.", "amber")
        st.dataframe(pd.DataFrame(result["path"]), use_container_width=True, hide_index=True)


def _case_desk(incident: dict, ready: int, total: int, models: list[dict], api_ready: bool, api_note: str) -> None:
    st.markdown(f'''
    <div class="case-slate">
      <div class="case-label">ACTIVE FILE <span class="status-chip">● REQUIRES EVIDENCE</span></div>
      <div class="case-name">{html.escape(incident['name'])}</div>
      <div class="case-meta"><span>CASE ID <b>{html.escape(incident['id'])}</b></span><span>OPENED <b>{html.escape(incident['opened'])}</b></span><span>OBSERVATION <b>{html.escape(incident['observed'])}</b></span></div>
    </div>''', unsafe_allow_html=True)
    runnable = sum(bool(m["ready"]) for m in models)
    cards = [
        _asset_card("CASE-LINKED EVIDENCE", "0 sources", "No AIS, SAR, or ocean-current inputs verified", "amber"),
        _asset_card("REPOSITORY ASSETS", f"{ready}/{total}", "Available files across configured roots", "aqua"),
        _asset_card("RUNNABLE ML MODELS", f"{runnable}/{len(models)}", "Model status checks current runtime dependencies", "red"),
        _asset_card("DEMO API", "ONLINE" if api_ready else "OFFLINE", api_note, "mint" if api_ready else "blue"),
    ]
    st.markdown(f'<div class="asset-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    left, right = st.columns([1.2, .8], gap="large")
    with left:
        st.markdown('<div class="section-head"><span>AREA OF INTEREST</span><span class="soft-tag">REFERENCE POINT · NOT VALIDATED</span></div>', unsafe_allow_html=True)
        _map(incident["lat"], incident["lon"])
        st.caption("The marker comes from project configuration; it is not a validated spill detection. Basemap tiles need an internet connection.")
    with right:
        st.markdown('<div class="section-head"><span>CASE READINESS</span><span class="soft-tag">LIVE FILE CHECK</span></div>', unsafe_allow_html=True)
        _card("No incident-linked evidence", "FUSION STATUS · ABSTAIN", "A SAR scene tied to this incident is missing. Vessel leads and spill-source conclusions are withheld.", "amber")
        for m in models:
            status = "READY FOR INFERENCE" if m["ready"] else "BLOCKED"
            tone = "mint" if m["ready"] else "red"
            st.markdown(f'<div class="model-row"><div><b>{html.escape(m["name"])}</b><div class="asset-note">{html.escape(m["note"])}</div></div><span class="model-state" style="color:{_C[tone]}">{status}</span></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-head"><span>QUICK ACTIONS</span></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Open evidence register", use_container_width=True, key="go_evidence"):
                st.session_state["desk_view"] = "Evidence register"
                st.rerun()
        with c2:
            if st.button("Run scenario lab", use_container_width=True, key="go_sim"):
                st.session_state["desk_view"] = "Scenario lab"
                st.rerun()


def _incident_view(incident: dict) -> None:
    st.markdown('<div class="section-head"><span>INCIDENT DOSSIER</span><span class="signal-tag">CONFIGURED REFERENCE · VERIFY BEFORE USE</span></div>', unsafe_allow_html=True)
    a, b = st.columns([.85, 1.15], gap="large")
    with a:
        _card("Case identity", "SOURCE · MARITRACE CONFIG", f"ID {html.escape(incident['id'])} · {html.escape(incident['name'])}. Open date: {html.escape(incident['opened'])}. Observation timestamp: {html.escape(incident['observed'])}.", "amber")
        st.markdown("#### Investigator working note")
        st.caption("Notes stay in this browser session and do not change scores or case decisions.")
        note = st.text_area("Working note", key="investigator_note", height=130, placeholder="Record a source, question, or next action…", label_visibility="collapsed")
        st.caption(f"{len(note)} characters · session only")
        _speech_tools()
    with b:
        st.markdown('<div class="section-head"><span>LOCATION CONTEXT</span><span class="soft-tag">CONFIGURED COORDINATES</span></div>', unsafe_allow_html=True)
        _map(incident["lat"], incident["lon"])
        st.caption(f"{incident['lat']:.4f}° N, {abs(incident['lon']):.4f}° W · no detection geometry loaded")


def _track_view() -> None:
    st.markdown('<div class="section-head"><span>TRACKS & DRIFT</span><span class="soft-tag">SOURCE AVAILABILITY CHECK</span></div>', unsafe_allow_html=True)
    traj = output_file("processed_data/vessel_trajectories.csv")
    drift = output_file("processed_data/drift_backward_trajectory.csv")
    if not traj:
        _card("No reconstructed vessel tracks", "TRACKS UNAVAILABLE", "The configured output roots contain no processed AIS trajectory file. This panel will populate when real trajectories are available.", "amber")
    else:
        df = pd.read_csv(traj, nrows=5000)
        coords = valid_coordinate_rows(df)
        st.metric("Track reports loaded", len(coords))
        st.dataframe(coords.head(100), use_container_width=True, hide_index=True)
    if not drift:
        _card("No incident drift solution", "CURRENTS FILE MISSING OR INVALID", "The local mirror lacks validated incident-specific drift output. Use the clearly marked simulation lab for demonstration only.", "amber")
    else:
        st.dataframe(pd.read_csv(drift, nrows=5000), use_container_width=True, hide_index=True)


def _model_lab(models: list[dict]) -> None:
    st.markdown('<div class="section-head"><span>MODEL BENCH</span><span class="signal-tag">ARTIFACT ≠ INFERENCE READINESS</span></div>', unsafe_allow_html=True)
    for m in models:
        tone = "mint" if m["ready"] else "red"
        _card(m["name"], "READY" if m["ready"] else "BLOCKED", html.escape(m["note"]), tone)
        st.caption(f"Artifact: {m['file'] if m['file'] else 'not found'}")
    image_path = data_file("sample/scene-1/sar_image.tif") or data_file("sample/scene-1/image.tif")
    with st.expander("Try SAR inference on a supplied image", expanded=False):
        model = next(m for m in models if m["kind"] == "vision")
        upload = st.file_uploader("SAR image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if not model["ready"]:
            st.warning("Inference is unavailable in this runtime. Install the pinned requirements into the project virtual environment, then restart the dashboard.")
        elif upload and st.button("Score this image", key="score_image"):
            try:
                net = load_satellite_model(model["file"])
                score = predict_satellite_score(net, Image.open(upload))
                st.metric("Model output (not validated incident probability)", f"{score:.3f}")
            except Exception as exc:
                st.error(f"Inference failed: {type(exc).__name__}: {exc}")
        if image_path:
            st.caption("A local sample image is available; local metrics are not case evidence.")
    vessel = next(m for m in models if m["kind"] == "tabular")
    st.markdown("#### Classify vessel activity from an AIS CSV")
    st.caption("Upload rows containing speed, course, distance_from_shore, distance_from_port, lat, lon, and timestamp (POSIX seconds). The first 5,000 rows are evaluated; output is gear class, not pollution responsibility.")
    ais_upload = st.file_uploader("AIS feature CSV", type=["csv"], key="vessel_ais_csv")
    if ais_upload:
        if not vessel["ready"]:
            st.warning(vessel["note"])
        else:
            try:
                raw = pd.read_csv(ais_upload, nrows=5000)
                required = {"speed", "course", "distance_from_shore", "distance_from_port", "lat", "lon", "timestamp"}
                missing = sorted(required - set(raw.columns))
                if missing:
                    st.error("Missing required columns: " + ", ".join(missing))
                elif raw.empty:
                    st.warning("The uploaded CSV has no rows.")
                else:
                    from .vessel_classifier import FEATURE_COLUMNS, GEAR_CLASSES, extract_kinematic_features
                    import joblib
                    model = joblib.load(vessel["file"])
                    features = extract_kinematic_features(raw)[FEATURE_COLUMNS]
                    labels = model.predict(features).astype(int)
                    probabilities = model.predict_proba(features)
                    scores = probabilities.max(axis=1)
                    output = raw.copy()
                    output["predicted_gear"] = [GEAR_CLASSES[i] if 0 <= i < len(GEAR_CLASSES) else "unknown" for i in labels]
                    output["class_score_uncalibrated"] = scores.round(3)
                    st.success(f"Classified {len(output):,} AIS rows. Scores are uncalibrated model outputs.")
                    st.dataframe(output.head(100), use_container_width=True, hide_index=True)
                    st.download_button("Download classified AIS CSV", output.to_csv(index=False), "maritrace_vessel_predictions.csv", "text/csv")
            except Exception as exc:
                st.error(f"Vessel inference failed: {type(exc).__name__}: {exc}")


def _evidence_register(rows: list[dict]) -> None:
    st.markdown('<div class="section-head"><span>EVIDENCE REGISTER</span><span class="soft-tag">LOCAL ARTIFACT INVENTORY</span></div>', unsafe_allow_html=True)
    df = pd.DataFrame(rows)
    if "Path" in df:
        df["Path"] = df["Path"].map(lambda v: Path(v).name if v and v != "—" else "—")
    col1, col2 = st.columns([.7, 1.3])
    with col1:
        st.metric("Available items", int((df["Status"] == "Available").sum()), f"of {len(df)} indexed")
    with col2:
        st.caption("Paths are shortened for readability. “Available” confirms only that a non-empty file exists; provenance and case linkage require separate verification.")
    states = ["All", *sorted(df["Status"].unique())]
    chosen = st.selectbox("Filter by availability", states, label_visibility="collapsed")
    visible = df if chosen == "All" else df[df["Status"] == chosen]
    st.dataframe(visible, use_container_width=True, hide_index=True, height=460)
    p1, p2 = st.columns(2)
    with p1:
        _card("Source coverage", "MISSING CASE DATA", "The project mirror has no NOAA incident CSV, processed AIS trajectories, audit log, or event-linked SAR scene.", "amber")
    with p2:
        _card("Time alignment", "BASELINE ONLY", "The January 2, 2024 AIS snapshot predates the configured January 7 incident opening; it cannot substantiate incident-time vessel activity.", "red")


def render_dashboard() -> None:
    st.markdown(f"""
    <style>
    :root{{--bg:{_C['bg']};--panel:{_C['panel']};--line:{_C['line']};--text:{_C['text']};--muted:{_C['muted']};--aqua:{_C['aqua']};}}
    .stApp{{background:var(--bg);color:var(--text);font-family:Inter,'Segoe UI',sans-serif}}
    [data-testid="stHeader"]{{background:var(--bg)}}
    [data-testid="stMainBlockContainer"]{{max-width:1580px;padding-top:1.05rem;padding-bottom:3rem}}
    [data-testid="stSidebar"]{{background:#0b1b23;border-right:1px solid #1c343d}}
    [data-testid="stSidebar"] *{{color:#dbe8e4}}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"]{{color:#86a09e}}
    [data-testid="stSidebar"] hr{{border-color:#28434b}}
    [data-testid="stSidebar"] [data-testid="stRadio"] label{{padding:.38rem .55rem;border-radius:7px}}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){{background:#19353b;color:#54e0cf}}
    .mast{{display:flex;align-items:center;justify-content:space-between;gap:14px;border-bottom:1px solid var(--line);padding:4px 0 17px;margin-bottom:20px}}
    .brand{{font-size:14px;letter-spacing:2.5px;font-weight:850;color:#e8f3ee}} .brand span{{color:var(--aqua)}}
    .mast-right{{font-size:11px;letter-spacing:1.1px;text-transform:uppercase;color:#93aaa7}}
    .live-dot{{display:inline-block;width:7px;height:7px;border-radius:50%;background:#ffbd66;margin-right:8px;box-shadow:0 0 12px #ffbd66}}
    .case-slate{{background:#10232c;border:1px solid #28434b;border-left:3px solid #54e0cf;border-radius:10px;padding:18px 21px;margin:2px 0 14px}}
    .case-label,.section-head{{display:flex;align-items:center;justify-content:space-between;gap:8px;color:#9bb0ae;font-size:10px;letter-spacing:1.55px;font-weight:800;text-transform:uppercase}}
    .case-name{{font-size:clamp(22px,2.5vw,31px);font-weight:760;letter-spacing:-.8px;color:#e6f0ec;margin:9px 0 12px}}
    .case-meta{{display:flex;gap:26px;flex-wrap:wrap;font-size:10px;letter-spacing:1px;color:#77918e}}
    .case-meta b{{color:#c5d7d2;letter-spacing:.3px;margin-left:5px;font-weight:650}}
    .status-chip,.soft-tag,.signal-tag{{border-radius:4px;padding:4px 7px;background:#382c21;color:#ffbd66;font-size:9px;letter-spacing:1px;font-weight:800}}
    .soft-tag{{background:#162a31;color:#9bb0ae}} .signal-tag{{background:#2d2620;color:#ffbd66}}
    .asset-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0 23px}}
    .asset-card{{background:#0d1e26;border:1px solid #1f3941;border-radius:8px;padding:13px 14px;min-height:84px}}
    .asset-label{{font-size:9px;letter-spacing:1px;font-weight:800;color:#86a09e}}
    .asset-value{{font-size:22px;font-weight:800;letter-spacing:-.4px;margin:5px 0 1px}}
    .asset-note{{font-size:11px;color:#91a7a3;line-height:1.45}}
    .section-head{{margin:5px 0 11px;padding-top:2px}}
    .surface,.model-row{{background:#10232c;border:1px solid #28434b;border-radius:9px;padding:15px 16px;margin:7px 0 13px}}
    .surface-title{{font-size:16px;font-weight:740;margin:6px 0;color:#e6f0ec}}
    .kicker{{font-size:9px;font-weight:850;letter-spacing:1.3px}}
    .surface-body{{font-size:12px;color:#a8bcba;line-height:1.55}}
    .model-row{{display:flex;align-items:center;justify-content:space-between;gap:12px}}
    .model-state{{font-size:9px;font-weight:850;letter-spacing:1px;white-space:nowrap}}
    .stApp h1,.stApp h2,.stApp h3,.stApp h4{{color:#e6f0ec!important;letter-spacing:-.025em}}
    .stApp p,.stApp label,.stApp [data-testid="stCaptionContainer"]{{color:#a8bcba}}
    [data-testid="stMetric"]{{background:#0d1e26;border:1px solid #28434b;border-radius:8px;padding:12px 14px}}
    [data-testid="stMetricLabel"]{{color:#9bb0ae}} [data-testid="stMetricValue"]{{color:#e6f0ec}}
    [data-testid="stDataFrame"]{{border:1px solid #28434b;border-radius:8px;overflow:hidden}}
    div.stButton>button,div.stDownloadButton>button{{background:#132a34;color:#cde0db;border:1px solid #36545b;border-radius:7px;font-weight:700;min-height:38px}}
    div.stButton>button[kind="primary"]{{background:#54e0cf;border-color:#54e0cf;color:#07141b}}
    div.stButton>button:hover{{border-color:#54e0cf;color:#54e0cf}}
    div.stButton>button[kind="primary"]:hover{{background:#76efdf;color:#07141b}}
    [data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-baseweb="textarea"]>div{{background:#0d1e26;color:#e6f0ec;border-color:#36545b;border-radius:7px}}
    textarea{{background:#0d1e26!important;color:#e6f0ec!important}}
    [data-testid="stAlert"]{{background:#221e18;border-color:#79562d}}
    [data-testid="stAlert"] *{{color:#f0d3a9!important}}
    hr{{border-color:#28434b}}
    @media(max-width:850px){{.asset-grid{{grid-template-columns:repeat(2,1fr)}}.case-meta{{gap:8px 16px}}.mast-right{{font-size:9px}}}}
    @media(max-width:550px){{.asset-grid{{grid-template-columns:1fr 1fr}}.case-slate{{padding:14px}}}}
    </style>
    <div class="mast"><div class="brand"><span>◉</span> MARITRACE <span style="color:#829895;font-size:10px;letter-spacing:1px">/ FORENSICS</span></div><div class="mast-right"><i class="live-dot"></i>LOCAL ANALYSIS ENVIRONMENT &nbsp;·&nbsp; DATA NOT LIVE</div></div>
    """, unsafe_allow_html=True)
    incident = _load_incident()
    ready, total = _records_available()
    models = _models()
    api_ready, api_note = _api_status()
    rows = inventory()
    with st.sidebar:
        st.markdown("<div style='font-size:10px;letter-spacing:1.7px;color:#78918e;font-weight:800;margin:7px 0 13px'>INVESTIGATION DESK</div>", unsafe_allow_html=True)
        labels = ["Operations deck", "Incident dossier", "Tracks & drift", "Model bench", "Evidence register", "Scenario lab"]
        icons = ["◈", "▣", "⌁", "◉", "▤", "▷"]
        choices = [f"{i}  {label}" for i, label in zip(icons, labels)]
        existing = st.session_state.get("desk_view", "Operations deck")
        current_idx = labels.index(existing) if existing in labels else 0
        selected = st.radio("Workspace", choices, index=current_idx, label_visibility="collapsed", key="desk_nav")
        view = labels[choices.index(selected)]
        st.session_state["desk_view"] = view
        st.markdown("---")
        st.markdown("<div style='font-size:9px;letter-spacing:1.4px;color:#78918e;font-weight:800'>SYSTEM READINESS</div>", unsafe_allow_html=True)
        st.caption(f"{ready} of {total} indexed artifacts present")
        st.caption(f"{sum(m['ready'] for m in models)} of {len(models)} ML models runnable")
        st.caption(f"Demo API · {'online' if api_ready else 'offline'}")
        st.markdown("---")
        st.caption("Build 1.0 · Session state is local to this browser tab.")
    if view == "Operations deck":
        _case_desk(incident, ready, total, models, api_ready, api_note)
    elif view == "Incident dossier":
        _incident_view(incident)
    elif view == "Tracks & drift":
        _track_view()
    elif view == "Model bench":
        _model_lab(models)
    elif view == "Evidence register":
        _evidence_register(rows)
    else:
        _demo_panel()
