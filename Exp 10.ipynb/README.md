# 🫁 Automated Chest X-Ray Pneumonia Detection System

An end-to-end Machine Learning Medical Diagnostics Application implementing a **single-frame web interface** that covers all 5 fundamental stages:
1. **Load Input Image:** Accepts custom patient chest radiographs (`.png`, `.jpg`, `.jpeg`) with instant built-in clinical demo presets (Normal vs. Pneumonia).
2. **Preprocessor:** Applies Grayscale Normalization, Median Noise Filtering, **CLAHE** (Contrast Limited Adaptive Histogram Equalization), and Otsu Thresholding for lung opacity segmentation.
3. **Feature Extraction:** Computes **GLCM Texture Descriptors** (*Contrast, Dissimilarity, Homogeneity, Energy, Correlation*), Statistical Intensity Moments (*Mean, Std, Skewness, Kurtosis*), and **Opacity Density Percentage**.
4. **Model Training & Prediction:** Uses a **Random Forest Classifier (120 Trees)** to classify Normal vs. Pneumonia with confidence scoring.
5. **Performance Evaluation:** Provides real-time **Accuracy, Precision, Recall (Sensitivity), F1-Score, ROC-AUC curve**, and an interactive **Confusion Matrix Heatmap**.

---

## 🚀 How to Run the Project in VS Code

### Option 1: Using the Terminal in VS Code
1. Open **Visual Studio Code**.
2. Go to **File ➔ Open Folder...** and select:
   ```
   C:\Users\HP\Downloads\Chest_XRay_Pneumonia_Detection
   ```
3. Open a new Terminal in VS Code (`Ctrl + ` `~` or **Terminal ➔ New Terminal**).
4. (Optional but recommended) Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
5. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
6. Start the web application:
   ```bash
   streamlit run app.py
   ```
7. Your default web browser will automatically open:
   ```
   http://localhost:8501
   ```

---

### Option 2: 1-Click Launch (Windows)
Simply double-click the [run.bat](file:///c:/Users/HP/Downloads/Chest_XRay_Pneumonia_Detection/run.bat) file inside the project folder. It will install all dependencies and launch the application in your browser.

---

## 📁 Project Directory Structure
```
Chest_XRay_Pneumonia_Detection/
│
├── app.py              # Main Streamlit web application (All 5 stages in one frame)
├── requirements.txt    # Python library dependencies
├── run.bat             # 1-Click Windows execution script
└── README.md           # Documentation & instructions
```
