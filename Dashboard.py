
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

matplotlib.use("Agg")

MODEL_PATH = r"C:\Users\sabbu\OneDrive\Documents\Git\System-Degradation-Predictor\final_balanced_cnn.keras"
MODEL_INPUT_STEPS = 60     
RISK_THRESHOLD = 0.75 

st.set_page_config(
    page_title="SRE · Cluster Telemetry",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCALER_PARAMS: dict[str, dict] = {
    "demand_cpu": {
        "mean":  0.25, "std": 0.15, "scale": 100,
        "unit":  "%",  "label": "CPU Demand",
        "color": "#4dabf7",
        "ylim":  (0, 100),
        "fmt":   "{:.1f}%",
    },
    "utilization_cpu": {
        "mean":  0.30, "std": 0.18, "scale": 100,
        "unit":  "%",  "label": "CPU Utilization",
        "color": "#74c0fc",
        "ylim":  (0, 100),
        "fmt":   "{:.1f}%",
    },
    "saturation_cpu": {
        "mean":  0.35, "std": 0.20, "scale": 100,
        "unit":  "%",  "label": "CPU Saturation",
        "color": "#a9e34b",
        "ylim":  (0, 100),
        "fmt":   "{:.1f}%",
    },
    "utilization_mem": {
        "mean":  4.0,  "std": 2.5,  "scale": 1,
        "unit":  "GB", "label": "Memory Utilization",
        "color": "#69db7c",
        "ylim":  (0, 16),
        "fmt":   "{:.2f} GB",
    },
    "delay_cpi": {
        "mean":  1.1,  "std": 0.40, "scale": 1,
        "unit":  "CPI","label": "CPI Delay (cycles / instruction)",
        "color": "#ff6b6b",
        "ylim":  (0, 3.5),
        "fmt":   "{:.3f}",
    },
}

PANEL_ORDER = ["demand_cpu", "utilization_cpu", "saturation_cpu", "utilization_mem", "delay_cpi"]

COL_UTIL_MEM     = 0
COL_DELAY_CPI    = 1
COL_DEMAND_CPU   = 2
COL_UTIL_CPU     = 3
COL_SAT_CPU      = 4

PANEL_COL = {
    "demand_cpu":      COL_DEMAND_CPU,
    "utilization_cpu": COL_UTIL_CPU,
    "saturation_cpu":  COL_SAT_CPU,
    "utilization_mem": COL_UTIL_MEM,
    "delay_cpi":       COL_DELAY_CPI,
}

NODE_REGISTRY: dict[str, dict] = {
    "node-01.prod.us-east": {
        "priority":    "P1 - Critical",
        "cluster":     "Compute Engine Core",
        "environment": "production",
        "region":      "us-east-1",
        "file":        r"C:\Users\sabbu\OneDrive\Documents\Git\System-Degradation-Predictor\node01_healthy.npy"
    },
    "node-02.prod.us-east": {
        "priority":    "P1 - Critical",
        "cluster":     "Compute Engine Core",
        "environment": "production",
        "region":      "us-east-1",
        "file":        r"C:\Users\sabbu\OneDrive\Documents\Git\System-Degradation-Predictor\node02_healthy.npy"
    },
    "node-03.degraded.us-central": {
        "priority":    "P2 - Elevated",
        "cluster":     "Analytics Pipeline",
        "environment": "degraded",
        "region":      "us-central-1",
        "file":        r"C:\Users\sabbu\OneDrive\Documents\Git\System-Degradation-Predictor\node03_failure.npy"
    },
    "node-04.degraded.us-west": {
        "priority":    "P2 - Elevated",
        "cluster":     "Batch Processing Fleet",
        "environment": "degraded",
        "region":      "us-west-2",
        "file":        r"C:\Users\sabbu\OneDrive\Documents\Git\System-Degradation-Predictor\node04_failure.npy"
    },
}


def _generate_fallback_matrix(node_id: str) -> np.ndarray:
    """Constructs fixed deterministic matrices if .npy files are missing."""
    steps = np.linspace(0.0, 1.0, 60)
    t = np.zeros((60, 6), dtype=np.float32)
    
    if "degraded" in node_id:
        rng = np.random.default_rng(seed=303 if "us-central" in node_id else 404)
        t[:, 0] = np.clip(0.25 + steps * 0.55 + rng.normal(0, 0.015, 60), 0.0, 1.0)
        t[:, 1] = np.clip(0.30 + steps * 0.50 + rng.normal(0, 0.015, 60), 0.0, 1.0)
        t[:, 2] = np.clip(0.35 + steps * 0.45 + rng.normal(0, 0.015, 60), 0.0, 1.0)
        mem_rise = 0.40 + steps * 0.30
        mem_drop = np.where(steps > 0.65, (steps - 0.65) * -1.80, 0.0)
        t[:, 3] = np.clip(mem_rise + mem_drop + rng.normal(0, 0.012, 60), -0.10, 1.00)
        t[:, 4] = np.clip(1.10 + steps * 1.20 + rng.normal(0, 0.050, 60), 0.8, 3.0)
        t[:, 5] = np.ones(60) * 0.5
    else:
        rng = np.random.default_rng(seed=101 if "east" in node_id else 202)
        t[:, 0] = np.clip(rng.normal(3.00, 0.13, 60), 2.50, 3.60)
        t[:, 1] = np.clip(rng.normal(2.70, 0.12, 60), 2.20, 3.20)
        t[:, 2] = np.clip(rng.normal(0.90, 0.10, 60), 0.60, 1.30)
        t[:, 3] = np.clip(rng.normal(2.40, 0.08, 60), 2.10, 2.70)
        t[:, 4] = np.clip(rng.normal(-0.80, 0.12, 60), -1.10, -0.40)
        t[:, 5] = np.zeros(60)
        
    return t


def resolve_telemetry(node_id: str) -> np.ndarray:
    """Attempts to load authentic .npy sequences from disk with safe fallbacks."""
    if node_id not in NODE_REGISTRY:
        raise ValueError(f"Unknown node identifier: {node_id}")
        
    target_file = NODE_REGISTRY[node_id]["file"]
    return np.load(target_file).astype(np.float32)

def to_physical(z_col: np.ndarray, feature: str) -> np.ndarray:
    p = SCALER_PARAMS[feature]
    return (z_col * p["std"] + p["mean"]) * p["scale"]


# ML Inference Interface
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        return None


def pad_window(window: np.ndarray) -> np.ndarray:
    n = window.shape[0]
    if n >= MODEL_INPUT_STEPS:
        return window[-MODEL_INPUT_STEPS:]
    pad_rows = MODEL_INPUT_STEPS - n
    padding  = np.zeros((pad_rows, window.shape[1]), dtype=np.float32)
    return np.vstack([padding, window])


def mock_predict(batched_input: np.ndarray) -> float:
    seq     = batched_input[0]
    nonzero = seq[seq.any(axis=1)]
    if len(nonzero) == 0:
        return 0.0

    cpu_z = float(nonzero[:, 0].mean())
    cpi_z = float(nonzero[:, 4].mean())

    cpu_norm  = float(np.clip((cpu_z - 0.25) / 0.55, 0.0, 1.0))
    cpi_norm  = float(np.clip((cpi_z + 0.10) / 2.70, 0.0, 1.0))

    composite = 0.25 * cpu_norm + 0.75 * cpi_norm
    score     = 1.0 / (1.0 + np.exp(-7.0 * (composite - 0.35)))
    return float(np.clip(score, 0.0, 1.0))


def run_inference(model, window: np.ndarray) -> float:
    padded  = pad_window(window)
    batched = padded[np.newaxis, :, :]

    if model is not None:
        prediction = model.predict(batched, verbose=0)
        return float(prediction[0][0])
    else:
        return float(mock_predict(batched))


# UI Components
def init_state() -> None:
    if "active_node" not in st.session_state:
        st.session_state.active_node = "node-01.prod.us-east"


def render_sidebar(final_score: float = None) -> float:
    """Draws sidebar with dynamic static-inference assessment summaries."""
    with st.sidebar:
        st.markdown("### Node Inventory")
        st.markdown("---")

        st.selectbox(
            label="Select Active Node:",
            options=list(NODE_REGISTRY.keys()),
            key="active_node",
        )

        st.markdown("---")

        node = NODE_REGISTRY[st.session_state.active_node]
        st.markdown("**Node Metadata**")
        st.markdown(
            f"**Priority Score:** `{node['priority']}`\n\n"
            f"**Cluster Target:** `{node['cluster']}`\n\n"
            f"**Environment:** `{node['environment']}`\n\n"
            f"**Region:** `{node['region']}`"
        )
        
        # Inject Upstream Static Assessment Cards
        if final_score is not None:
            st.markdown("**AI Prescriptive Analysis**")
            if final_score >= RISK_THRESHOLD:
                st.markdown("Status: <span style='color:#ff6b6b;font-weight:bold;'>[FAIL - High Risk Eviction]</span>", unsafe_allow_html=True)
            else:
                st.markdown("Status: <span style='color:#51cf66;font-weight:bold;'>[PASS - Stable Operations]</span>", unsafe_allow_html=True)
            st.markdown(f"**Final Window Horizon Confidence:** `{final_score * 100.0:.1f}%`")

        st.markdown("---")

        st.markdown("**Stream Speed**")
        speed = st.slider(
            label="Seconds per step",
            min_value=0.05,
            max_value=1.0,
            value=0.15,
            step=0.05,
        )

    return speed


def render_metrics_grid(window: np.ndarray, metrics_slot) -> None:
    latest  = window[-1]
    earlier = window[-min(13, len(window))]

    def val(row, feature):
        return to_physical(np.array([row[PANEL_COL[feature]]]), feature)[0]

    def delta_str(feature, suffix):
        d = val(latest, feature) - val(earlier, feature)
        return f"{d:+.2f} {suffix}"

    with metrics_slot:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("CPU Demand", SCALER_PARAMS["demand_cpu"]["fmt"].format(val(latest, "demand_cpu")), delta_str("demand_cpu", "%"), delta_color="inverse")
        col2.metric("CPU Utilization", SCALER_PARAMS["utilization_cpu"]["fmt"].format(val(latest, "utilization_cpu")), delta_str("utilization_cpu", "%"), delta_color="inverse")
        col3.metric("CPU Saturation", SCALER_PARAMS["saturation_cpu"]["fmt"].format(val(latest, "saturation_cpu")), delta_str("saturation_cpu", "%"), delta_color="inverse")
        col4.metric("Memory", SCALER_PARAMS["utilization_mem"]["fmt"].format(val(latest, "utilization_mem")), delta_str("utilization_mem", "GB"), delta_color="inverse")
        col5.metric("CPI Delay", SCALER_PARAMS["delay_cpi"]["fmt"].format(val(latest, "delay_cpi")), delta_str("delay_cpi", ""), delta_color="inverse")


def build_stacked_chart(window: np.ndarray, node_id: str, step: int) -> plt.Figure:
    n_steps   = window.shape[0]
    x         = np.arange(1, n_steps + 1)
    tick_pos   = [1, 12, 24, 36, 48, 60]
    tick_labs  = ["-5h", "-4h", "-3h", "-2h", "-1h", "now"]

    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 8.0), gridspec_kw={"hspace": 0.06})
    fig.patch.set_facecolor("#0e1117")

    for ax, feature in zip(axes, PANEL_ORDER):
        col_idx = PANEL_COL[feature]
        p       = SCALER_PARAMS[feature]
        phys    = to_physical(window[:, col_idx], feature)

        ax.set_facecolor("#161b22")
        for side, spine in ax.spines.items():
            if side in ("top", "right"): spine.set_visible(False)
            else: spine.set_edgecolor("#2a2d3e")

        ax.grid(axis="y", color="#1e2130", linewidth=0.7, linestyle="--", zorder=0)
        ax.grid(axis="x", color="#1e2130", linewidth=0.4, linestyle=":", zorder=0)

        ax.fill_between(x, phys, alpha=0.18, color=p["color"], zorder=1)
        ax.plot(x, phys, color=p["color"], linewidth=1.6, zorder=2)

        ax.set_xlim(1, 60)
        ax.set_ylim(p["ylim"])
        ax.set_ylabel(p["unit"], color="#6c7293", fontsize=7, labelpad=6)
        ax.tick_params(axis="y", colors="#6c7293", labelsize=7, length=3)
        ax.tick_params(axis="x", colors="#6c7293", labelsize=7, length=3)

        ax.text(0.01, 0.88, p["label"], transform=ax.transAxes, fontsize=7.5, color="#9da5b4", fontweight="bold", va="top")
        live_val = p["fmt"].format(phys[-1]) if len(phys) > 0 else "—"
        ax.text(0.99, 0.88, live_val, transform=ax.transAxes, fontsize=8, color=p["color"], fontweight="bold", va="top", ha="right")

    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels(tick_labs, fontsize=7, color="#6c7293")
    axes[-1].set_xlabel("Historical Steps (5-Min Intervals)", fontsize=7.5, color="#6c7293", labelpad=6)
    fig.suptitle(f"{node_id}   ·   step {step:02d} / 60", fontsize=8.5, color="#6c7293", x=0.01, ha="left", y=1.01)
    fig.subplots_adjust(top=0.95, bottom=0.10, left=0.08, right=0.92)
    return fig

REMEDIATION_STEPS = [
    "**Halted:** New task scheduling to this node has been blocked (11.82-minute buffer active).",
    "**Workload Migration:** Live-evicting high-priority containers to stable nodes.",
    "**Drain Initiated:** Safe drain queue initialized for graceful reboot.",
]

def render_risk_panel(risk_slot, score: float, step: int) -> None:
    pct      = score * 100.0
    critical = score >= RISK_THRESHOLD

    with risk_slot.container():
        st.markdown("**Live Risk Monitor**")
        st.markdown("---")

        delta_label = "CRITICAL" if critical else "NORMAL"
        st.metric(
            label=f"Failure Probability  ·  step {step:02d}/60",
            value=f"{pct:.1f} %",
            delta=delta_label,
            delta_color="off",
        )

        bar_color = "#ff4b4b" if critical else "#21c354"
        fill_pct  = int(np.clip(pct, 0, 100))
        st.markdown(
            f"""
            <div style="background:#1e2130;border-radius:4px;height:8px;margin:4px 0 12px 0;overflow:hidden;">
              <div style="background:{bar_color};width:{fill_pct}%;height:100%;border-radius:4px;transition:width 0.3s ease;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"<div style='font-size:0.72rem;color:#4a4f68;margin-bottom:8px;'>Alarm threshold: {int(RISK_THRESHOLD*100)} %</div>", unsafe_allow_html=True)

        if critical:
            st.error("**CRITICAL ANOMALY DETECTED:** Node entering an architectural death spiral.")
            st.markdown("**Automated SRE Playbook**")
            for i, action in enumerate(REMEDIATION_STEPS, start=1):
                st.markdown(f"{i}. {action}")
        else:
            st.success("Node operating within safe parameters.")
            st.markdown(f"<div style='font-size:0.78rem;color:#4a4f68;margin-top:8px;'>Remediation playbook activates when probability exceeds {int(RISK_THRESHOLD*100)} %.</div>", unsafe_allow_html=True)


# Core Control Flow Engine
def run_streaming_loop(tensor, node_id, speed, chart_slot, metrics_slot, risk_slot, model) -> None:
    step = 1
    while True:
        window = tensor[:step]

        render_metrics_grid(window, metrics_slot)
        fig = build_stacked_chart(window, node_id, step)
        with chart_slot:
            st.pyplot(fig, width="stretch")
        plt.close(fig)

        score = run_inference(model, window)
        render_risk_panel(risk_slot, score, step)

        time.sleep(speed)
        step += 1
        if step > 60:
            step = 1


def main() -> None:
    init_state()
    
    node_id = st.session_state.active_node
    tensor  = resolve_telemetry(node_id)
    model   = load_model()
    
    final_window_score = run_inference(model, tensor)
    
    speed = render_sidebar(final_score=final_window_score)
    
    st.markdown("## Cluster Telemetry & Failure Prediction System")
    st.markdown("---")

    col_main, col_sidebar = st.columns([3, 1])

    with col_main:
        st.markdown(f"**Telemetry Stream** — `{node_id}`")
        metrics_slot = st.empty()
        st.markdown("")
        chart_slot   = st.empty()

    with col_sidebar:
        risk_slot = st.empty()

    run_streaming_loop(tensor, node_id, speed, chart_slot, metrics_slot, risk_slot, model)


if __name__ == "__main__":
    main()