import streamlit as st
import os
import cv2
from PIL import Image
import numpy as np
import requests

# Import backend modules
from backend.inference import load_aetherdx_model, run_inference, preprocess_image
from backend.gradcam import generate_gradcam, save_and_display_gradcam
from backend.explanation import get_explanation
from backend.compliance import generate_decision_id, get_current_timestamp, evaluate_compliance_evidence, generate_audit_report_text

# -- Streamlit Page Config --
st.set_page_config(
    page_title="XAI-COMPASS - AI Explainability Platform",
    page_icon="🧭",
    layout="centered"
)

# -- Model Selection via Sidebar --
st.sidebar.title("Configuration")
st.sidebar.markdown("Select an existing model to connect to the XAI-COMPASS platform.")

# List models in the models directory
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
available_models = [f for f in os.listdir(models_dir) if f.endswith(".keras") or f.endswith(".h5")]

if not available_models:
    st.sidebar.error("No models found in the 'models' folder. Please add a .keras or .h5 file.")
    st.stop()

selected_model_file = st.sidebar.selectbox("Choose AI Model", available_models)

# -- Caching Model Loading --
@st.cache_resource
def load_model(model_filename):
    model_path = os.path.join(models_dir, model_filename)
    return load_aetherdx_model(model_path)

try:
    model = load_model(selected_model_file)
except Exception as e:
    st.error(f"Failed to load model '{selected_model_file}': {e}")
    st.stop()

# -- UI Header --
st.title("XAI-COMPASS ● System Ready")
st.subheader("Multilingual AI Explainability & Compliance Platform")
st.write("Make AI decisions understandable, multilingual and audit-ready.")
st.markdown("**Explain • Translate • Verify • Audit**")
st.markdown("---")

# -- Simple Architecture Visual --
st.markdown("""
<div style='text-align: center; color: gray; font-size: 14px;'>
EXISTING MODEL &rarr; MODEL CONNECTOR &rarr; XAI-COMPASS &rarr; EXPLAIN &rarr; TRANSLATE &rarr; VERIFY &rarr; AUDIT<br>
<i>Current demonstration model: AetherDx InceptionV3</i>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# -- API Health Check --
api_url = "http://127.0.0.1:8000"
api_status_text = "🔴 Disconnected"
try:
    health_response = requests.get(f"{api_url}/health", timeout=2)
    if health_response.status_code == 200:
        api_status_text = "● Connected"
except Exception:
    st.error("XAI-COMPASS API is unavailable. Please start the Model Connector.\n\n`py -3.13 -m uvicorn backend.api:app --reload --port 8000`")
    st.stop()

# -- Model Connector Card --
st.markdown("### CONNECTED AI MODEL")
st.info(f"""
**Model:** {selected_model_file}  
**Domain:** Healthcare  
**Integration:** Existing AI Model via API  
**API Status:** {api_status_text}  
""")
st.caption("Designed as a governance layer around existing AI/ML models. AetherDx is the current healthcare reference implementation.")

# -- Future Models Section --
with st.expander("DESIGNED FOR EXISTING AI/ML MODELS"):
    st.markdown("""
    **Current:**  
    ✓ AetherDx Healthcare Model  
    
    **Future:**  
    ○ Medical Imaging Models  
    ○ Fraud Detection Models  
    ○ Credit Risk Models  
    ○ Insurance Risk Models  
    """)

st.markdown("---")

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

# -- Image Upload Section --
st.markdown("### MODEL INPUT")
st.write("Current reference implementation: nail-image healthcare screening.")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    if st.button("ANALYZE", type="primary", use_container_width=True):
        st.session_state.decision_id = generate_decision_id()
        st.session_state.timestamp = get_current_timestamp()
        st.session_state.analysis_done = True
        st.session_state.total_decisions += 1
        
if uploaded_file is not None and st.session_state.analysis_done:
    with st.spinner("Analyzing image and generating explanation..."):
        try:
            # Save uploaded file temporarily
            temp_dir = "outputs"
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, "temp_upload.jpg")
            
            image = Image.open(uploaded_file).convert("RGB")
            image.save(temp_image_path)
            
            # 1. Run API Inference (Primary Prediction)
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
            
            # Use API response for AI decision
            prediction = api_result.get("display_prediction", api_result.get("prediction", "Unknown"))
            
            confidence = api_result.get("probability", 0.0)
            confidence_percent = f"{confidence * 100:.2f}"
            
            st.markdown("---")
            
            # 2. Display AI Decision
            st.markdown("### AI DECISION")
            st.markdown(f"**Prediction:** {prediction}")
            st.markdown(f"**Model Output Probability:** ~{confidence_percent}%")
            st.caption("Note: Model output probability does not represent medical certainty.")
            st.markdown(f"**Model:** {selected_model_file}")
            st.markdown("**Domain:** Healthcare")
            st.markdown(f"**Decision ID:** {st.session_state.decision_id}")
            
            # 3. Explainable AI Section
            st.markdown("---")
            st.markdown("### WHY DID THE MODEL MAKE THIS DECISION?")
            
            gradcam_success = False
            xai_status = "Failed"
            try:
                xai_dir = os.path.join("outputs", "xai")
                os.makedirs(xai_dir, exist_ok=True)
                cam_path = os.path.join(xai_dir, "gradcam_overlay.jpg")
                
                preprocessed_img_array = preprocess_image(temp_image_path)
                heatmap, _ = generate_gradcam(model, preprocessed_img_array, layer_name="mixed10")
                save_and_display_gradcam(temp_image_path, heatmap, cam_path=cam_path, alpha=0.4)
                
                original_img = Image.open(temp_image_path)
                overlay_img = Image.open(cam_path)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_img, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(overlay_img, caption="Grad-CAM Evidence", use_container_width=True)
                    
                st.info("The highlighted regions indicate image areas that contributed most strongly to the model's prediction.")
                gradcam_success = True
                xai_status = "Generated"
                
                # Update Monitoring metric
                st.session_state.xai_generated = max(st.session_state.xai_generated, st.session_state.total_decisions)
                
            except Exception as xai_e:
                st.warning("XAI visualization temporarily unavailable.")
                st.error(f"Grad-CAM Error: {str(xai_e)}")
            
            # 4. Multilingual Explanation
            st.markdown("---")
            st.markdown("### MULTILINGUAL EXPLANATION")
            st.write("Convert technical AI output into understandable explanations for diverse users.")
            
            tab_en, tab_hi, tab_mr = st.tabs(["🇬🇧 English", "🇮🇳 हिंदी", "🇮🇳 मराठी"])
            
            en_exp = get_explanation("english", prediction, confidence_percent)
            hi_exp = get_explanation("hindi", prediction, confidence_percent)
            mr_exp = get_explanation("marathi", prediction, confidence_percent)
            
            with tab_en:
                st.write(en_exp)
            with tab_hi:
                st.write(hi_exp)
            with tab_mr:
                st.write(mr_exp)
                
            # 5. AI Governance & Compliance Evidence
            st.markdown("---")
            st.markdown("### AI GOVERNANCE & COMPLIANCE EVIDENCE")
            
            checklist, passed_checks, total_checks, score = evaluate_compliance_evidence(
                has_prediction=True,
                has_gradcam=gradcam_success,
                has_explanation=True
            )
            
            for check, status in checklist.items():
                if status:
                    st.markdown(f"🟢 **{check}**")
                else:
                    st.markdown(f"🔴 **{check}**")
                    
            st.markdown(f"#### {passed_checks}/{total_checks} Evidence Checks Passed")
            st.markdown(f"## Evidence Readiness: {score}%")
            st.caption("This score indicates the availability of configured explainability and audit evidence. It does not constitute legal or regulatory certification.")
            
            # Update Evidence metric if all passed
            if score == 100:
                st.session_state.evidence_complete = max(st.session_state.evidence_complete, st.session_state.total_decisions)
                
            # 6. Decision Monitoring Panel
            st.markdown("---")
            st.markdown("### DECISION MONITORING")
            st.markdown(f"**Total Decisions:** {st.session_state.total_decisions}")
            st.markdown(f"**XAI Generated:** {st.session_state.xai_generated}")
            st.markdown(f"**Evidence Complete:** {st.session_state.evidence_complete}")
            st.markdown("**Human Review:** Required")
            
            # 7. Audit Section
            st.markdown("---")
            st.markdown("### AUDIT-READY DECISION RECORD")
            
            # Displaying audit info on screen
            st.markdown(f"**Decision ID:** {st.session_state.decision_id}")
            st.markdown(f"**Timestamp:** {st.session_state.timestamp}")
            st.markdown(f"**Model:** {selected_model_file}")
            st.markdown(f"**Prediction:** {prediction}")
            st.markdown(f"**Model Output Probability:** ~{confidence_percent}%")
            st.markdown("**XAI Method:** Grad-CAM")
            st.markdown("**XAI Evidence:** Original Input, Grad-CAM Heatmap, Grad-CAM Overlay")
            st.markdown(f"**Language:** English (Default for report)")
            st.markdown("**Compliance Evidence:** Available")
            
            # Create report text
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
                label="GENERATE AUDIT REPORT",
                data=report_text,
                file_name=f"XAI_COMPASS_Audit_{st.session_state.decision_id}.txt",
                mime="text/plain",
                type="primary",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
        finally:
            if os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                except:
                    pass

# -- Medical Disclaimer --
st.markdown("---")
st.caption("AI-assisted screening demonstration. This output is not a medical diagnosis and should not replace evaluation by a qualified healthcare professional.")
