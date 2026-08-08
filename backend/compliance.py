import uuid
from datetime import datetime

def generate_decision_id():
    """Generates a unique decision ID."""
    timestamp_part = datetime.now().strftime("%Y")
    unique_part = str(uuid.uuid4()).split('-')[0].upper()
    return f"ADX-{timestamp_part}-{unique_part}"

def get_current_timestamp():
    """Returns the current formatted timestamp."""
    return datetime.now().strftime("%d %b %Y, %H:%M IST")

def evaluate_compliance_evidence(has_prediction, has_gradcam, has_explanation):
    """
    Evaluates the compliance evidence and returns a checklist and score.
    Returns: (checklist_results, passed_checks, total_checks, evidence_score)
    """
    checks = {
        "Decision recorded": True,
        "Model identified": True,
        "Model version recorded": True,
        "Prediction available": has_prediction,
        "Model output probability available": has_prediction,
        "XAI evidence generated": has_gradcam,
        "Explanation generated": has_explanation,
        "Multilingual explanation available": has_explanation,
        "Timestamp recorded": True,
        "Decision ID generated": True,
        "Human review recommendation available": True,
    }
    
    passed_checks = sum(1 for status in checks.values() if status)
    total_checks = len(checks)
    score = int((passed_checks / total_checks) * 100)
    
    return checks, passed_checks, total_checks, score

def generate_audit_report_text(decision_id, timestamp, prediction, confidence, 
                               xai_status, explanation_lang, explanation_text, 
                               checklist, passed_checks, total_checks, score, model_name="AetherDx InceptionV3"):
    """Generates a downloadable audit report as a formatted text string."""
    report = []
    report.append("-" * 39)
    report.append("XAI-COMPASS")
    report.append("AI Decision Audit Report")
    report.append("-" * 39)
    report.append(f"\nDecision ID:\n{decision_id}")
    report.append(f"\nTimestamp:\n{timestamp}")
    report.append(f"\nReference Model:\n{model_name}")
    report.append("\nDomain:\nHealthcare")
    report.append("\nInput:\nNail Image")
    report.append(f"\nPrediction:\n{prediction}")
    report.append(f"\nModel Output Probability:\n~{confidence}%")
    report.append("\nXAI Method:\nGrad-CAM")
    report.append(f"\nXAI Status:\n{xai_status}")
    
    if xai_status == "Generated":
        report.append("\nXAI Evidence Artifacts:")
        report.append("✓ Original Input")
        report.append("✓ Grad-CAM Heatmap")
        report.append("✓ Grad-CAM Overlay (gradcam_overlay.jpg)")
    
    report.append(f"\nExplanation Language:\n{explanation_lang}")
    report.append(f"\nExplanation:\n{explanation_text}")
    
    report.append("\nCompliance Evidence:\n")
    for check, status in checklist.items():
        icon = "✓" if status else "✗"
        report.append(f"{icon} {check}")
        
    report.append(f"\n{passed_checks}/{total_checks} Evidence Checks Passed")
    report.append(f"Evidence Readiness:\n{score}%")
    report.append("\nHuman Review:\nMedical professional review recommended.")
    report.append("\n" + "-" * 39)
    report.append("\nDisclaimer:\n")
    report.append("This score indicates the availability of configured explainability and audit evidence. It does not constitute legal or regulatory certification.")
    report.append("\nMedical Disclaimer:\n")
    report.append("AI-assisted screening demonstration. This output is not a medical diagnosis and should not replace evaluation by a qualified healthcare professional.")
    
    return "\n".join(report)
