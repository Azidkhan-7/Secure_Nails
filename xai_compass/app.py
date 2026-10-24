import streamlit as st
import os
import sys
import cv2
from PIL import Image
import numpy as np
import requests

# Ensure repository root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import backend modules
from aetherdx.inference.inference import load_aetherdx_model, run_inference, preprocess_image
from xai_compass.xai.gradcam import generate_gradcam, save_and_display_gradcam
from xai_compass.explanation.explanation import get_explanation
from xai_compass.governance.compliance import generate_decision_id, get_current_timestamp, evaluate_compliance_evidence, generate_audit_report_text

# -- Streamlit Page Config --
st.set_page_config(
    page_title="XAI-COMPASS - AI Explainability Platform",
    page_icon="🧭",
    layout="wide"
)

# -- Custom CSS --
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .api-badge-connected { background-color: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 9999px; font-size: 0.875rem; font-weight: 600; display: inline-block; border: 1px solid #bbf7d0; }
    .api-badge-disconnected { background-color: #fee2e2; color: #991b1b; padding: 4px 12px; border-radius: 9999px; font-size: 0.875rem; font-weight: 600; display: inline-block; border: 1px solid #fecaca; }
    .decision-card { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 32px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-top: 1rem; }
    .prediction-title { font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 600; margin-bottom: 0.5rem; }
    .prediction-value { font-size: 2.5rem; font-weight: 800; color: #0f172a; margin-bottom: 0.25rem; line-height: 1.2; }
    .probability-value { font-size: 1.25rem; color: #3b82f6; font-weight: 600; }
    .model-info-text { font-size: 0.875rem; color: #94a3b8; margin-top: 1.5rem; }
    .section-header { font-size: 1.125rem; font-weight: 700; color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; margin-top: 2rem; margin-bottom: 1.5rem; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

# -- API Health Check --
api_url = "http://127.0.0.1:8000"
api_status_text = "API Connected"
api_badge_class = "api-badge-connected"
try:
    health_response = requests.get(f"{api_url}/health", timeout=2)
    if health_response.status_code != 200:
        api_status_text = "API Error"
        api_badge_class = "api-badge-disconnected"
except Exception:
    api_status_text = "API Unavailable"
    api_badge_class = "api-badge-disconnected"

# -- Sidebar (Modernized) --
st.sidebar.markdown(f"<h2>XAI-COMPASS</h2>", unsafe_allow_html=True)
st.sidebar.markdown(f"<div class='{api_badge_class}'>● {api_status_text}</div><br>", unsafe_allow_html=True)
st.sidebar.markdown("---")

models_dir = os.path.join(ROOT_DIR, "aetherdx", "model")
if not os.path.exists(models_dir) or not any(f.endswith(".keras") or f.endswith(".h5") for f in os.listdir(models_dir)):
    models_alt = os.path.join(ROOT_DIR, "models")
    if os.path.exists(models_alt) and any(f.endswith(".keras") or f.endswith(".h5") for f in os.listdir(models_alt)):
        models_dir = models_alt
os.makedirs(models_dir, exist_ok=True)
available_models = [f for f in os.listdir(models_dir) if f.endswith(".keras") or f.endswith(".h5")]

if not available_models:
    st.sidebar.error("No models found in the 'models' folder. Please add a .keras or .h5 file.")
    st.stop()

st.sidebar.markdown("**CONFIGURATION**")
selected_model_file = st.sidebar.selectbox("Model Selector", available_models, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("**INTEGRATION DETAILS**")
st.sidebar.caption(f"**Model:** {selected_model_file}")
st.sidebar.caption("**Domain:** Healthcare")
st.sidebar.caption("**Integration:** Existing AI Model via API")
st.sidebar.caption("**XAI Engine:** Grad-CAM + LLM Templates")

# -- Caching Model Loading --
@st.cache_resource
def load_model(model_filename):
    model_path = os.path.join(models_dir, model_filename)
    return load_aetherdx_model(model_path)

try:
    model = load_model(selected_model_file)
except Exception as e:
    st.error(f"Failed to load local model '{selected_model_file}' for XAI: {e}")
    st.stop()

if api_status_text == "API Unavailable":
    st.error("XAI-COMPASS API is unavailable. Please start the Model Connector.\n\npy -3.13 -m uvicorn xai_compass.api.api:app --reload --port 8000")
    st.stop()

# -- Main Header --
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown("<h1 style='margin-bottom: 0; padding-bottom: 0;'>XAI-COMPASS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.25rem; color: #475569; margin-top: 0;'>Multilingual Explainable AI & Compliance Platform</p>", unsafe_allow_html=True)

# -- Session State Initialization --
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'decision_id' not in st.session_state:
    st.session_state.decision_id = ""
if 'timestamp' not in st.session_state:
    st.session_state.timestamp = ""
if 'total_decisions' not in st.session_state:
    st.session_state.total_decisions = 0
if 'xai_generated' not in st.session_state:
    st.session_state.xai_generated = 0
if 'evidence_complete' not in st.session_state:
    st.session_state.evidence_complete = 0

# -- Main Layout --
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("<div class='section-header'>MODEL INPUT</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        st.image(uploaded_file, use_container_width=True, caption="Target Image")
        if st.button("ANALYZE IMAGE", type="primary", use_container_width=True):
            st.session_state.decision_id = generate_decision_id()
            st.session_state.timestamp = get_current_timestamp()
            st.session_state.analysis_done = True
            st.session_state.total_decisions += 1

with col_right:
    st.markdown("<div class='section-header'>AI DECISION</div>", unsafe_allow_html=True)
    
    if not uploaded_file or not st.session_state.analysis_done:
        st.info("Upload an image and click 'ANALYZE IMAGE' to generate a prediction and compliance report.")
    else:
        with st.spinner("Analyzing via API..."):
            temp_dir = "outputs"
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, "temp_upload.jpg")
            
            image = Image.open(uploaded_file).convert("RGB")
            image.save(temp_image_path)
            
            # API Inference
            try:
                with open(temp_image_path, "rb") as f:
                    response = requests.post(
                        f"{api_url}/predict",
                        files={"file": (os.path.basename(temp_image_path), f, "image/jpeg")},
                        timeout=30
                    )
                response.raise_for_status()
                api_result = response.json()
            except requests.exceptions.RequestException as req_err:
                st.error(f"API Prediction Request Failed: {req_err}")
                st.stop()
            
            prediction = api_result.get("display_prediction", api_result.get("prediction", "Unknown"))
            confidence = api_result.get("probability", 0.0)
            confidence_percent = f"{confidence * 100:.2f}"
            
            # Render Decision Card
            st.markdown(f"""
            <div class='decision-card'>
                <div class='prediction-title'>PREDICTED CLASS</div>
                <div class='prediction-value'>{prediction}</div>
                <div class='probability-value'>~{confidence_percent}% Probability</div>
                <div class='model-info-text'>
                    Model: AetherDx InceptionV3 | Decision ID: {st.session_state.decision_id}<br>
                    Note: Model output probability does not represent medical certainty.
                </div>
            </div>
            """, unsafe_allow_html=True)

if uploaded_file is not None and st.session_state.analysis_done:
    
    st.markdown("<div class='section-header'>EXPLAINABILITY EVIDENCE</div>", unsafe_allow_html=True)
    
    gradcam_success = False
    xai_status = "Failed"
    
    col_xai1, col_xai2 = st.columns([1, 1], gap="large")
    
    with col_xai1:
        st.markdown("**VISUAL EVIDENCE (GRAD-CAM)**")
        try:
            xai_dir = os.path.join("outputs", "xai")
            os.makedirs(xai_dir, exist_ok=True)
            cam_path = os.path.join(xai_dir, "gradcam_overlay.jpg")
            
            preprocessed_img_array = preprocess_image(temp_image_path)
            heatmap, _ = generate_gradcam(model, preprocessed_img_array, layer_name="mixed10")
            save_and_display_gradcam(temp_image_path, heatmap, cam_path=cam_path, alpha=0.4)
            
            overlay_img = Image.open(cam_path)
            
            st.image(overlay_img, use_container_width=True, caption="Model Attention Map (Heatmap Overlay)")
            st.caption("Highlighted regions indicate image areas that contributed most strongly to the model's decision.")
            
            gradcam_success = True
            xai_status = "Generated"
            st.session_state.xai_generated = max(st.session_state.xai_generated, st.session_state.total_decisions)
            
        except Exception as xai_e:
            st.error(f"XAI visualization failed: {str(xai_e)}")

    with col_xai2:
        st.markdown("**MULTILINGUAL EXPLANATION**")
        st.caption("AI output translated for diverse users.")
        
        tab_en, tab_hi, tab_mr = st.tabs(["🇬🇧 English", "🇮🇳 हिंदी", "🇮🇳 मराठी"])
        
        en_exp = get_explanation("english", prediction, confidence_percent)
        hi_exp = get_explanation("hindi", prediction, confidence_percent)
        mr_exp = get_explanation("marathi", prediction, confidence_percent)
        
        with tab_en:
            st.info(en_exp)
        with tab_hi:
            st.info(hi_exp)
        with tab_mr:
            st.info(mr_exp)
            
    st.markdown("<div class='section-header'>AI GOVERNANCE & AUDIT PLATFORM</div>", unsafe_allow_html=True)
    
    checklist, passed_checks, total_checks, score = evaluate_compliance_evidence(
        has_prediction=True,
        has_gradcam=gradcam_success,
        has_explanation=True
    )
    
    if score == 100:
        st.session_state.evidence_complete = max(st.session_state.evidence_complete, st.session_state.total_decisions)
        
    col_gov1, col_gov2 = st.columns([2, 1], gap="large")
    
    with col_gov1:
        st.markdown("**COMPLIANCE CHECKLIST**")
        
        # Grid layout for checklist (2 columns)
        chk_cols = st.columns(2)
        items = list(checklist.items())
        half = len(items) // 2 + len(items) % 2
        
        for i, (check, status) in enumerate(items):
            col_idx = 0 if i < half else 1
            icon = "🟢" if status else "🔴"
            with chk_cols[col_idx]:
                st.markdown(f"{icon} {check}")
                
    with col_gov2:
        st.markdown("**EVIDENCE COMPLETENESS**")
        st.metric(label="Compliance Score", value=f"{score}%", delta=f"{passed_checks}/{total_checks} Checks Passed" if score==100 else f"-{total_checks-passed_checks} Missing", delta_color="normal" if score==100 else "inverse")
        st.caption("Indicates availability of configured explainability evidence. Not a regulatory certification.")
        
    st.markdown("---")
    
    col_audit1, col_audit2 = st.columns([1, 1], gap="large")
    
    with col_audit1:
        st.markdown("**DECISION MONITORING**")
        mon_col1, mon_col2, mon_col3 = st.columns(3)
        with mon_col1:
            st.metric("Total Decisions", st.session_state.total_decisions)
        with mon_col2:
            st.metric("XAI Generated", st.session_state.xai_generated)
        with mon_col3:
            st.metric("Evidence Complete", st.session_state.evidence_complete)
            
    with col_audit2:
        st.markdown("**AUDIT-READY DECISION RECORD**")
        st.markdown(f"**ID:** {st.session_state.decision_id} | **Time:** {st.session_state.timestamp}")
        st.markdown(f"**AI Decision:** {prediction} (~{confidence_percent}%)")
        st.markdown("**XAI Method:** Grad-CAM")
        
        report_text = generate_audit_report_text(
            decision_id=st.session_state.decision_id,
            timestamp=st.session_state.timestamp,
            prediction=prediction,
            confidence=confidence_percent,
            xai_status=xai_status,
            explanation_lang="English",
            explanation_text=en_exp,
            checklist=checklist,
            passed_checks=passed_checks,
            total_checks=total_checks,
            score=score,
            model_name=selected_model_file
        )
        
        st.download_button(
            label="📄 GENERATE AUDIT REPORT",
            data=report_text,
            file_name=f"XAI_COMPASS_Audit_{st.session_state.decision_id}.txt",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )

# -- Medical Disclaimer --
st.markdown("---")
st.caption("AI-assisted screening demonstration. This output is not a medical diagnosis and should not replace evaluation by a qualified healthcare professional.")
