import os
import json
import pandas as pd
import subprocess
import modal
import streamlit as st

# ----------------------------------------------------------------------------
# Volume and Image Setup
# ----------------------------------------------------------------------------
metrics_volume = modal.Volume.from_name("evaluation_dashboard_results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("streamlit", "pandas", "plotly")
    .add_local_file(__file__, "/root/serve_dashboard.py")
)

# ----------------------------------------------------------------------------
# Define and Deploy the App
# ----------------------------------------------------------------------------
app = modal.App(
    name="metrics-dashboard",
    image=image,
    volumes={"/data": metrics_volume},
)


@app.function()
@modal.web_server(port=8000)
def run():
    """
    Launch Streamlit server on port 8000 inside the container.
    """
    cmd = (
        "streamlit run /root/serve_dashboard.py "
        "--server.port 8000 "
        "--server.enableCORS=false "
        "--server.enableXsrfProtection=false"
    )
    subprocess.Popen(cmd, shell=True)


# ----------------------------------------------------------------------------
# Streamlit Dashboard Application Logic
# ----------------------------------------------------------------------------
def main():
    metrics_volume.reload()

    st.set_page_config(page_title="Metrics Dashboard", layout="wide")
    st.title("📊 Cognee Evaluations Dashboard")

    data_path = "/data"
    records = []

    for filename in sorted(os.listdir(data_path)):
        if not filename.endswith(".json"):
            continue
        base = filename.rsplit(".", 1)[0]
        parts = base.split("_")
        benchmark = parts[1] if len(parts) >= 3 else ""

        full_path = os.path.join(data_path, filename)
        with open(full_path, "r") as f:
            items = json.load(f)
        num_q = len(items)
        total_em = sum(q["metrics"]["EM"]["score"] for q in items)
        total_f1 = sum(q["metrics"]["f1"]["score"] for q in items)
        total_corr = sum(q["metrics"]["correctness"]["score"] for q in items)
        records.append(
            {
                "file": parts[0].upper() + "_____" + parts[2],
                "benchmark": benchmark,
                "questions": num_q,
                "avg_EM": round(total_em / num_q, 4),
                "avg_F1": round(total_f1 / num_q, 4),
                "avg_correctness": round(total_corr / num_q, 4),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        st.warning("No JSON files found in the volume.")
        return

    st.subheader("Results by benchmark")
    for bm, group in df.groupby("benchmark"):
        st.markdown(f"### {bm}")
        st.dataframe(
            group[["file", "questions", "avg_EM", "avg_F1", "avg_correctness"]],
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
