import io
import base64
from pathlib import Path

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vial Delamination Detection",
    page_icon="🔬",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .result-banner-ok   { background:#f0fdf4; color:#15803d; border:1.5px solid #86efac;
                        border-radius:10px; padding:12px 16px; font-weight:700; font-size:1rem; margin-bottom:8px; }
  .result-banner-warn { background:#fff7ed; color:#c2410c; border:1.5px solid #fdba74;
                        border-radius:10px; padding:12px 16px; font-weight:700; font-size:1rem; margin-bottom:8px; }
  .stat-chip { background:#f8fafc; border:1px solid #e2e8f0; border-radius:9px;
               padding:10px 14px; text-align:center; margin-bottom:6px; }
  .stat-chip .num  { font-size:1.3rem; font-weight:700; color:#1a202c; }
  .stat-chip .lbl  { font-size:0.65rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.4px; }
  .rc-tag { display:inline-block; background:#f1f5f9; border-radius:5px;
            padding:2px 8px; font-size:0.7rem; color:#2563eb; font-weight:500; margin:2px; }
  .section-label { font-size:0.68rem; font-weight:700; color:#94a3b8;
                   text-transform:uppercase; letter-spacing:.8px; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)

# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_model():
    model_path = Path(__file__).parent / "best .pt"
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}\nPlease place 'best .pt' in the same folder as this app.")
        st.stop()
    return YOLO(str(model_path))

model = load_model()

# ── Helpers ────────────────────────────────────────────────────────────────────
def is_delaminated(detections: list[dict]) -> bool:
    return any(
        "delaminated" in d["class"].lower() and "non" not in d["class"].lower()
        for d in detections
    )

def run_detection(img: Image.Image, conf: float, iou: float):
    results = model.predict(source=np.array(img), conf=conf, iou=iou, verbose=False)
    r = results[0]
    annotated = Image.fromarray(r.plot()[:, :, ::-1])
    detections = [
        {"class": model.names[int(b.cls)], "conf": round(float(b.conf), 4)}
        for b in r.boxes
    ]
    return annotated, detections

def render_banner(detections: list[dict]):
    if is_delaminated(detections):
        st.markdown('<div class="result-banner-warn">⚠️ Delaminated Vial</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-banner-ok">✅ Non-Delaminated Vial</div>', unsafe_allow_html=True)

def render_stats(detections: list[dict]):
    n = len(detections)
    uniq = len({d["class"] for d in detections})
    max_conf = max((d["conf"] for d in detections), default=0)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-chip"><div class="num">{n}</div><div class="lbl">Detections</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-chip"><div class="num">{uniq}</div><div class="lbl">Classes</div></div>', unsafe_allow_html=True)
    with c3:
        conf_str = f"{max_conf*100:.0f}%" if n else "—"
        st.markdown(f'<div class="stat-chip"><div class="num">{conf_str}</div><div class="lbl">Max Conf</div></div>', unsafe_allow_html=True)

def render_detections(detections: list[dict]):
    st.markdown('<div class="section-label">Defect List</div>', unsafe_allow_html=True)
    if not detections:
        st.success("✅ All clear — no defects found.")
    else:
        for d in detections:
            pct = d["conf"] * 100
            col_name, col_conf, col_bar = st.columns([3, 1, 2])
            with col_name:
                st.write(f"**{d['class']}**")
            with col_conf:
                st.write(f"{pct:.1f}%")
            with col_bar:
                st.progress(d["conf"])

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🔬 Vial Delamination Detection")
st.caption("Upload vial images and run YOLO-based delamination analysis.")
st.divider()

# ── Sidebar settings ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    conf = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    iou  = st.slider("IoU Threshold",        0.05, 0.95, 0.45, 0.05)
    st.divider()
    st.caption("🔬 Vial Delamination Detection\nPowered by YOLOv8 + Streamlit")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_single, tab_multi = st.tabs(["📷  Single Image", "🗂️  Multi Image"])

# ════════════════════════════════════════════════════════════════════════════════
# SINGLE IMAGE TAB
# ════════════════════════════════════════════════════════════════════════════════
with tab_single:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("#### Upload Image")
        uploaded = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            key="single_upload",
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption=uploaded.name, use_container_width=True)
            run_btn = st.button("⚙️  Run Detection", key="single_run", use_container_width=True, type="primary")
        else:
            st.info("📂 Drop or browse an image to get started.")
            run_btn = False

    with right:
        st.markdown("#### Detection Result")
        if uploaded and run_btn:
            with st.spinner("Running detection…"):
                annotated, detections = run_detection(img, conf, iou)
            st.image(annotated, caption="Annotated Result", use_container_width=True)
            render_banner(detections)
            render_stats(detections)
            render_detections(detections)

            # Download button
            buf = io.BytesIO()
            annotated.save(buf, format="JPEG", quality=90)
            st.download_button(
                "⬇️  Download Result",
                data=buf.getvalue(),
                file_name=f"result_{uploaded.name}",
                mime="image/jpeg",
                use_container_width=True,
            )
        elif not uploaded:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#94a3b8;">
              <div style="font-size:3rem;">🧪</div>
              <p><strong style="color:#475569;">No Results Yet</strong></p>
              <p>Upload an image and click <strong>Run Detection</strong> to see results here.</p>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# MULTI IMAGE TAB
# ════════════════════════════════════════════════════════════════════════════════
with tab_multi:
    left_m, right_m = st.columns([1, 2], gap="large")

    with left_m:
        st.markdown("#### Upload Images")
        uploaded_files = st.file_uploader(
            "Choose multiple images",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            accept_multiple_files=True,
            key="multi_upload",
        )
        if uploaded_files:
            st.caption(f"**{len(uploaded_files)} file(s) selected**")
            for f in uploaded_files:
                st.markdown(f"📷 {f.name}")
            run_multi_btn = st.button("⚙️  Run Multi Detection", key="multi_run", use_container_width=True, type="primary")
        else:
            st.info("📂 Drop or browse multiple images to get started.")
            run_multi_btn = False

    with right_m:
        st.markdown("#### Multi-Image Detection Results")

        if uploaded_files and run_multi_btn:
            all_results = []
            progress_bar = st.progress(0, text="Processing images…")

            for i, f in enumerate(uploaded_files):
                img_m = Image.open(f).convert("RGB")
                ann, dets = run_detection(img_m, conf, iou)
                all_results.append({"filename": f.name, "image": ann, "detections": dets})
                progress_bar.progress((i + 1) / len(uploaded_files), text=f"Processed {i+1}/{len(uploaded_files)}")

            progress_bar.empty()

            # Summary chips
            total    = len(all_results)
            total_d  = sum(len(r["detections"]) for r in all_results)
            defects  = sum(1 for r in all_results if is_delaminated(r["detections"]))
            clean    = total - defects

            c1, c2, c3, c4 = st.columns(4)
            for col, num, lbl, color in [
                (c1, total,   "Images",        "#1a202c"),
                (c2, total_d, "Detections",    "#1a202c"),
                (c3, defects, "Delaminated",   "#c2410c"),
                (c4, clean,   "Non-Delaminated","#15803d"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="stat-chip"><div class="num" style="color:{color}">{num}</div>'
                        f'<div class="lbl">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.divider()

            # Result cards — 2 per row
            for i in range(0, len(all_results), 2):
                cols = st.columns(2, gap="medium")
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(all_results):
                        break
                    r = all_results[idx]
                    with col:
                        st.image(r["image"], caption=r["filename"], use_container_width=True)
                        n = len(r["detections"])
                        if is_delaminated(r["detections"]):
                            st.markdown('<span style="background:#fff7ed;color:#c2410c;border:1px solid #fdba74;border-radius:20px;padding:2px 10px;font-size:.75rem;font-weight:700;">⚠ Defect</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span style="background:#f0fdf4;color:#15803d;border:1px solid #86efac;border-radius:20px;padding:2px 10px;font-size:.75rem;font-weight:700;">✓ Clean</span>', unsafe_allow_html=True)

                        tags = " ".join(
                            f'<span class="rc-tag">{d["class"]} {d["conf"]*100:.0f}%</span>'
                            for d in r["detections"][:4]
                        )
                        if not tags:
                            tags = '<span style="font-size:.7rem;color:#94a3b8">No detections</span>'
                        st.markdown(tags, unsafe_allow_html=True)
                        st.caption(f"{n} detection{'s' if n != 1 else ''}")

                        # Per-image download
                        buf = io.BytesIO()
                        r["image"].save(buf, format="JPEG", quality=90)
                        st.download_button(
                            "⬇️ Download",
                            data=buf.getvalue(),
                            file_name=f"result_{r['filename']}",
                            mime="image/jpeg",
                            key=f"dl_{idx}",
                            use_container_width=True,
                        )
        elif not uploaded_files:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#94a3b8;">
              <div style="font-size:3rem;">🧪</div>
              <p><strong style="color:#475569;">No Results Yet</strong></p>
              <p>Upload images and click <strong>Run Multi Detection</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
