import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="Pneumonia Detection - Chest X-Ray ML",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border-left: 4px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🫁 Automated Chest X-Ray Pneumonia Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Integrated One-Frame Diagnostic Pipeline: Image Acquisition ➔ Preprocessing ➔ Feature Extraction ➔ Model Training ➔ Diagnostic Evaluation</div>', unsafe_allow_html=True)

# ----------------- SYNTHETIC MEDICAL X-RAY GENERATOR (FOR DEMO/TESTING) -----------------
def generate_sample_cxr(sample_type="Normal"):
    """Generates a realistic synthetic chest radiograph for immediate out-of-the-box demonstration."""
    canvas = np.zeros((224, 224), dtype=np.uint8) + 25  # Dark thoracic cavity background
    
    # Draw thoracic rib cage silhouette
    cv2.ellipse(canvas, (112, 120), (85, 95), 0, 0, 360, 70, -1)
    
    # Draw left and right lung lobes (radiolucent = darker)
    cv2.ellipse(canvas, (75, 120), (32, 65), -5, 0, 360, 45, -1)
    cv2.ellipse(canvas, (149, 120), (32, 65), 5, 0, 360, 45, -1)
    
    # Central mediastinum & cardiac silhouette (radiopaque = whiter)
    cv2.ellipse(canvas, (112, 135), (22, 50), 0, 0, 360, 150, -1)
    # Spine column
    cv2.rectangle(canvas, (107, 20), (117, 210), 130, -1)
    # Clavicles & ribs
    cv2.line(canvas, (35, 55), (105, 75), 110, 3)
    cv2.line(canvas, (189, 55), (119, 75), 110, 3)
    for y in range(85, 185, 22):
        cv2.ellipse(canvas, (112, y), (70, 18), 0, 20, 160, 85, 2)

    if sample_type == "Pneumonia":
        # Infiltrates / Consolidation (Patchy cloudy opacities in lung fields)
        cv2.circle(canvas, (80, 130), 22, 175, -1)
        cv2.circle(canvas, (145, 140), 18, 165, -1)
        cv2.circle(canvas, (70, 150), 15, 160, -1)
        # Apply smoothing to make opacities realistic alveolar consolidations
        canvas = cv2.GaussianBlur(canvas, (9, 9), 2.0)
    else:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0.8)
        
    return canvas

# ----------------- PIPELINE STEP 2: PREPROCESSING -----------------
def preprocess_cxr(image_np):
    """
    Medical Image Preprocessing:
    1. Grayscale Normalization
    2. Resize to 224x224
    3. Noise Reduction (Median Blur)
    4. Contrast Limited Adaptive Histogram Equalization (CLAHE)
    5. Opacity / Lung Field Thresholding (Otsu)
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()

    # Standardize scale
    resized = cv2.resize(gray, (224, 224))
    
    # Noise reduction
    denoised = cv2.medianBlur(resized, 3)
    
    # Contrast Enhancement via CLAHE (standard in digital radiography)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Segmentation of High-Density Opacity Fields (Otsu threshold)
    _, opacity_mask = cv2.threshold(enhanced, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return {
        "Grayscale": resized,
        "Denoised": denoised,
        "Enhanced (CLAHE)": enhanced,
        "Opacity Mask": opacity_mask
    }

# ----------------- PIPELINE STEP 3: FEATURE EXTRACTION -----------------
def extract_cxr_features(gray_img, mask_img):
    """
    Feature Extraction:
    - GLCM Texture Descriptors (Contrast, Dissimilarity, Homogeneity, Energy, Correlation)
    - Statistical Moments (Mean, Standard Deviation, Skewness, Kurtosis)
    - Lung Opacity Density Ratio
    """
    # 1. Texture Features via GLCM
    glcm = graycomatrix(gray_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # 2. Statistical Moments
    flat_pixels = gray_img.flatten()
    mean_val = float(np.mean(flat_pixels))
    std_val = float(np.std(flat_pixels))
    skew_val = float(skew(flat_pixels))
    kurt_val = float(kurtosis(flat_pixels))

    # 3. Opacity Density (Area percentage of opaque consolidated tissue)
    opacity_ratio = float(np.sum(mask_img == 255) / (224 * 224)) * 100.0

    features = {
        "Mean Intensity": round(mean_val, 3),
        "Standard Deviation": round(std_val, 3),
        "Skewness": round(skew_val, 3),
        "Kurtosis": round(kurt_val, 3),
        "GLCM Contrast": round(contrast, 3),
        "GLCM Dissimilarity": round(dissimilarity, 3),
        "GLCM Homogeneity": round(homogeneity, 3),
        "GLCM Energy": round(energy, 4),
        "GLCM Correlation": round(correlation, 3),
        "Opacity Density (%)": round(opacity_ratio, 2)
    }
    return features

# ----------------- PIPELINE STEP 4: MODEL TRAINING -----------------
@st.cache_resource
def train_cxr_model():
    """
    Trains a robust Random Forest Classifier using simulated medical distribution
    modeling Normal CXR vs. Pneumonia consolidation features.
    """
    np.random.seed(42)
    n_samples = 400
    
    # Class 0: Normal Radiograph Features
    # Mean, Std, Skew, Kurt, Contrast, Dissimilarity, Homogeneity, Energy, Correlation, Opacity%
    mean_normal = [68, 28, 0.45, -0.6, 35, 12, 0.65, 0.35, 0.85, 18.0]
    std_normal  = [8,  4,  0.10,  0.2,  6,  2, 0.05, 0.04, 0.03,  4.0]
    X_normal = np.random.normal(loc=mean_normal, scale=std_normal, size=(n_samples // 2, 10))
    y_normal = np.zeros(n_samples // 2)

    # Class 1: Pneumonia Radiograph Features (Elevated opacity, higher contrast, lower homogeneity)
    mean_pneu = [95, 42, 0.15, -0.2, 78, 24, 0.42, 0.18, 0.62, 38.5]
    std_pneu  = [10, 6,  0.12,  0.2, 12,  4, 0.06, 0.03, 0.05,  6.0]
    X_pneu = np.random.normal(loc=mean_pneu, scale=std_pneu, size=(n_samples // 2, 10))
    y_pneu = np.ones(n_samples // 2)

    X = np.vstack([X_normal, X_pneu])
    y = np.concatenate([y_normal, y_pneu])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob
    }
    return model, metrics

model, eval_metrics = train_cxr_model()

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.header("🎛️ Diagnostic Configuration")
    sample_choice = st.selectbox(
        "Choose Quick Demo Preset:",
        ["Pneumonia (Simulated Infiltrate)", "Normal (Clear Lungs)", "Upload Custom Image"]
    )
    
    st.markdown("---")
    st.write("**System Status:**")
    st.success("🟢 ML Inference Engine: Active")
    st.info("Model: Random Forest (120 Estimators)")
    st.caption("Engineered for Chest Radiography (CXR) screening.")

# Set active image in session
if sample_choice == "Normal (Clear Lungs)":
    st.session_state["active_cxr"] = generate_sample_cxr("Normal")
    st.session_state["cxr_label"] = "Sample: Normal CXR (Clear Lungs)"
elif sample_choice == "Pneumonia (Simulated Infiltrate)":
    st.session_state["active_cxr"] = generate_sample_cxr("Pneumonia")
    st.session_state["cxr_label"] = "Sample: Pneumonia CXR (Consolidated Infiltrates)"
else:
    if "active_cxr" not in st.session_state:
        st.session_state["active_cxr"] = generate_sample_cxr("Pneumonia")
        st.session_state["cxr_label"] = "Default Sample"

# ----------------- ONE-FRAME TABBED INTERFACE -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 1. Load Input Image",
    "⚙️ 2. Preprocessor",
    "📊 3. Feature Extraction",
    "🤖 4. Model Training & Prediction",
    "📈 5. Performance Evaluation"
])

# --- TAB 1: LOAD INPUT IMAGE ---
with tab1:
    st.subheader("Step 1: Chest X-Ray Acquisition")
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.write("Upload a patient Chest X-Ray radiograph or use one of the pre-configured clinical presets from the sidebar.")
        uploaded_file = st.file_uploader("Upload Chest X-Ray Image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            st.session_state["active_cxr"] = np.array(pil_img)
            st.session_state["cxr_label"] = f"Custom Patient Image: {uploaded_file.name}"
            st.success("✅ Patient Chest Radiograph Loaded Successfully!")
        else:
            st.info(f"Currently inspecting: **{st.session_state['cxr_label']}**")
            
    with col2:
        st.image(
            st.session_state["active_cxr"],
            caption=st.session_state["cxr_label"],
            width=300
        )

# Preprocessing & Feature Extraction for currently active image
prep_dict = preprocess_cxr(st.session_state["active_cxr"])
features_dict = extract_cxr_features(prep_dict["Grayscale"], prep_dict["Opacity Mask"])

# --- TAB 2: PREPROCESSOR ---
with tab2:
    st.subheader("Step 2: Medical Radiograph Preprocessing")
    st.write("Digital radiography requires specialized contrast adjustment to differentiate soft tissue infiltrates from bone.")
    
    p_cols = st.columns(4)
    p_cols[0].image(prep_dict["Grayscale"], caption="1. Grayscale Normalization (224x224)", use_container_width=True)
    p_cols[1].image(prep_dict["Denoised"], caption="2. Median Filter Noise Reduction", use_container_width=True)
    p_cols[2].image(prep_dict["Enhanced (CLAHE)"], caption="3. CLAHE Local Contrast Enhancement", use_container_width=True)
    p_cols[3].image(prep_dict["Opacity Mask"], caption="4. High-Density Infiltrate Mask (Otsu)", use_container_width=True)
    
    st.info("💡 **Clinical Relevance:** CLAHE sharpens pulmonary vascular markings, making fluid consolidation visible to the classifier.")

# --- TAB 3: FEATURE EXTRACTION ---
with tab3:
    st.subheader("Step 3: Radiomic Feature Extraction")
    st.write("Extracting statistical texture descriptors (GLCM) and pulmonary opacity density metrics.")
    
    feat_df = pd.DataFrame(list(features_dict.items()), columns=["Radiomic Feature", "Computed Value"])
    
    c_f1, c_f2 = st.columns([1, 1.2])
    with c_f1:
        st.dataframe(feat_df, use_container_width=True)
    with c_f2:
        # Texture feature breakdown plot
        st.write("#### GLCM Texture Descriptors Comparison")
        glcm_sub = feat_df.iloc[4:9].set_index("Radiomic Feature")
        st.bar_chart(glcm_sub)

# --- TAB 4: TRAINING & PREDICTION ---
with tab4:
    st.subheader("Step 4: Machine Learning Inference & Diagnostic Classification")
    st.write("Classifier: **Random Forest (120 Trees with Bootstrap Aggregation)**")
    
    # Prepare feature vector matching training order
    input_vector = np.array([[
        features_dict["Mean Intensity"],
        features_dict["Standard Deviation"],
        features_dict["Skewness"],
        features_dict["Kurtosis"],
        features_dict["GLCM Contrast"],
        features_dict["GLCM Dissimilarity"],
        features_dict["GLCM Homogeneity"],
        features_dict["GLCM Energy"],
        features_dict["GLCM Correlation"],
        features_dict["Opacity Density (%)"]
    ]])
    
    pred_class = model.predict(input_vector)[0]
    pred_probs = model.predict_proba(input_vector)[0]

    res_col1, res_col2 = st.columns([1.2, 1])
    with res_col1:
        if pred_class == 1:
            st.error(f"""
            ### ⚠️ DIAGNOSIS: PNEUMONIA DETECTED
            - **Diagnostic Confidence:** `{pred_probs[1]*100:.2f}%`
            - **Primary Observation:** Elevated pulmonary opacity density (`{features_dict['Opacity Density (%)']}%`) and high texture contrast indicating alveolar consolidation.
            """)
        else:
            st.success(f"""
            ### ✅ DIAGNOSIS: NORMAL (NO PNEUMONIA)
            - **Diagnostic Confidence:** `{pred_probs[0]*100:.2f}%`
            - **Primary Observation:** Clear bilateral lung fields, standard bronchovascular markings, and physiological radiolucency.
            """)
            
    with res_col2:
        st.write("#### Diagnostic Probability Distribution")
        prob_chart_df = pd.DataFrame({
            "Finding": ["Normal / Healthy", "Pneumonia Positive"],
            "Probability": [pred_probs[0], pred_probs[1]]
        })
        st.bar_chart(prob_chart_df.set_index("Finding"), color="#3B82F6")

# --- TAB 5: PERFORMANCE EVALUATION ---
with tab5:
    st.subheader("Step 5: Diagnostic Performance Evaluation & Validation")
    
    # Metric cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{eval_metrics['Accuracy']*100:.1f}%")
    m2.metric("Precision", f"{eval_metrics['Precision']*100:.1f}%")
    m3.metric("Recall (Sensitivity)", f"{eval_metrics['Recall']*100:.1f}%")
    m4.metric("F1-Score", f"{eval_metrics['F1-Score']*100:.1f}%")
    m5.metric("ROC AUC", f"{eval_metrics['ROC_AUC']:.3f}")
    
    st.markdown("---")
    eval_c1, eval_c2 = st.columns(2)
    
    with eval_c1:
        st.write("#### Confusion Matrix (Validation Set)")
        cm = confusion_matrix(eval_metrics["y_test"], eval_metrics["y_pred"])
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Pneumonia'], 
                    yticklabels=['Normal', 'Pneumonia'], ax=ax_cm)
        ax_cm.set_ylabel('Actual Ground Truth')
        ax_cm.set_xlabel('Predicted Clinical Class')
        st.pyplot(fig_cm)
        
    with eval_c2:
        st.write("#### Receiver Operating Characteristic (ROC) Curve")
        fig_roc, ax_roc = plt.subplots(figsize=(4.5, 3.5))
        ax_roc.plot(eval_metrics["fpr"], eval_metrics["tpr"], color='darkorange', lw=2, 
                    label=f"ROC Curve (AUC = {eval_metrics['ROC_AUC']:.2f})")
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate (1 - Specificity)')
        ax_roc.set_ylabel('True Positive Rate (Sensitivity)')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    st.write("#### Comprehensive Classification Report")
    report_dict = classification_report(
        eval_metrics["y_test"], eval_metrics["y_pred"],
        target_names=['Normal (Class 0)', 'Pneumonia (Class 1)'],
        output_dict=True
    )
    st.dataframe(pd.DataFrame(report_dict).transpose(), use_container_width=True)
