# 🧬 RedCell AI: Detecting Blood Doping with AI-Powered RBC Morphology

RedCell AI is a computer vision and data science pipeline built to detect potential blood doping in athletes by analyzing red blood cell (RBC) morphology from blood smear images.

---

## 🚀 Project Overview

RedCell AI automates the process of:
- Detecting and segmenting individual RBCs from high-resolution blood smear images
- Extracting morphological features like dual diameters
- Classifying RBCs into microcyte, macrocyte, normal, or anomalous
- Detecting intra-individual variations over time using rolling statistical analysis
- Performing altitude-aware anomaly detection based on erythropoietic behavior

This project blends computer vision, medical image analysis, and sports physiology for advanced anti-doping detection.

---

## 📁 Folder Structure

```
RedCell_AI/
├── images/                  # Input blood smear images (raw)
├── cropped_rbc/            # Cropped RBCs after detection and segmentation
├── categorized_rbc/        # RBCs sorted into macrocyte/microcyte/normal/other
├── models/                 # Trained classification models (.pth)
├── scripts/                # Python scripts for detection, classification, analysis
├── results/                # Output visualizations, CSVs, reports
└── README.md               # Project documentation
```

---

## 🧠 Key Components

### 1. RBC Detection & Segmentation
- Image preprocessing with Otsu thresholding and morphological ops
- Watershed segmentation to separate overlapping RBCs

### 2. Feature Extraction
- Extract two longest diameters using convex hull distance metrics
- Resize and save each RBC image (64x64 px standard)

### 3. Classification
- Use a trained PyTorch model to classify each RBC
- Save class labels back into the master CSV

### 4. Anomaly Detection
- Normalize pixel diameters using microscope scale (e.g., 1 px = 0.1432 μm)
- Perform rolling mean/std z-score check over time
- Detect anomalous years based on RBC count/size per altitude level

---

## 📊 CSV Structure

Main CSV used for tracking metadata:

```csv
image,rbc_index,Diameter1,Diameter2,saved_file,class,year,altitude
image-118.png,0,48.7,49.1,img118_rbc0.png,normal,2025,Sea Level
```

Additional columns like `avg_diameter`, `rolling_mean`, `z_score`, `anomaly` are appended in later stages.

---

## 📈 Intra-Individual Variation Detection

Implemented using:
- Rolling mean and std over past 10 cells
- Z-score thresholding (`|z| > 2`) for anomaly
- Slide-level aggregation to detect yearly anomalies

This models each athlete’s personal baseline, avoiding population-wide generalizations.

---

## 🏔 Altitude-Based RBC Count Analysis

Checks if RBC count and diameter fall within expected ranges based on:
- Sea Level, Moderate, High Altitude
- Gender-specific physiological norms

This ensures natural hematological variation isn’t mistaken for manipulation.

---

## 🧩 Architecture

📌 <img width="601" alt="Screenshot 2025-05-02 at 12 32 33 PM" src="https://github.com/user-attachments/assets/87f89d42-c03f-4186-ae94-25990e1adbb0" />


_Recommended: Flowchart showing preprocessing → segmentation → cropping → classification → diameter calc → normalization → anomaly detection_

---

## 💬 Sample Output

```bash
🚨 Anomaly Detected in RBC measurements for year 2025.
✅ RBC count is within the expected range.
⚠️ RBC size is outside normal range at high altitude.
```

---

## 🛠 Technologies Used
- Python, OpenCV, NumPy, pandas, matplotlib
- PyTorch (for classification model)
- Streamlit (optional UI wrapper)
- Jupyter Notebooks (for prototyping)

---

## 🧪 Future Improvements
- Improve overlapping cell separation with deep learning (e.g., U-Net)
- Track per-athlete ID across years for tighter profiling
- Build a web dashboard using Streamlit for full interaction
- Integrate with ABP (Athlete Biological Passport) databases

---

## 🙋‍♂️ Author

**Vighnesh M S**  
AI Researcher & Developer  
Email: [vighnesh21ms@gmail.com](mailto:vighnesh21ms@gmail.com)

---

## 📄 License

This project is under MIT License.
