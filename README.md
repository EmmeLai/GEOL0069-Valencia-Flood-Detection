# 🌊 Valencia Flood Detection using Sentinel-1 SAR and AI
### GEOL0069 AI for Earth Observation — Final Assessment
**UCL Department of Earth Sciences**

---

##  Problem Description

On **29 October 2024**, a DANA (Depresión Aislada en Niveles Altos) isolated cold-drop weather system triggered catastrophic flash floods across the Valencia region of Spain. In just 8 hours, parts of Valencia received over 400mm of rainfall — more than a full year's average. The floods killed **237 people**, inundated more than **53,000 hectares** of land, and caused an estimated **€10.7 billion** in damages, making it one of the deadliest natural disasters in modern Spanish history.

Rapid and accurate flood mapping is critical for emergency response, rescue operations, and damage assessment. Traditional optical satellite imagery is often unusable during flood events because storm clouds block the view. This project addresses that problem by using **Synthetic Aperture Radar (SAR)** imagery from the **Sentinel-1A** satellite, which can see through clouds and rain.

This project demonstrates how AI and machine learning can be combined with free satellite data to automatically detect and map flooded areas within hours of a disaster.

---

## 🛰️ Remote Sensing Technique: Sentinel-1 SAR

**Why SAR?**

SAR (Synthetic Aperture Radar) is an active remote sensing technique that emits its own microwave radar pulses and measures the energy reflected back from the Earth's surface. Unlike optical sensors, SAR does not depend on sunlight and penetrates clouds, making it uniquely valuable for flood monitoring during storms.

**How SAR detects floods:**

Water surfaces are very smooth compared to land. When radar pulses hit flat water, they bounce away from the satellite (specular reflection), returning almost no energy. This makes flooded areas appear **very dark** in SAR imagery. In contrast, rough surfaces like vegetation and urban areas scatter radar energy back to the satellite and appear **bright**.

**Data used:**
| Image | Date | Satellite | Mode | Polarisations |
|---|---|---|---|---|
| Pre-flood | 19 October 2024 | Sentinel-1A | IW (Interferometric Wide) | VV, VH |
| Post-flood | 31 October 2024 | Sentinel-1A | IW (Interferometric Wide) | VV, VH |

All data was accessed for free via **Google Earth Engine (GEE)** — no downloading required.

---

## 🤖 AI Methods

This project implements three AI/ML techniques from the GEOL0069 course:

### 1. K-means Clustering (Unsupervised Learning — Week 3/4)
K-means is an unsupervised algorithm that groups pixels into clusters based on their SAR backscatter values, without requiring any labelled training data. We used 3 clusters representing water/flood, vegetation, and urban areas. The water cluster was identified as the one with the lowest post-flood VV backscatter value — physically consistent with SAR behaviour over water surfaces.

### 2. Random Forest Classification (Supervised Learning — Week 2)
Random Forest is an ensemble of decision trees that votes on the class of each pixel. We used the K-means cluster labels as pseudo-labels to train the Random Forest on 6 input features (pre_VV, pre_VH, post_VV, post_VH, diff_VV, diff_VH), then applied the trained model to classify every pixel in the Valencia region.

### 3. Explainable AI — Permutation Feature Importance (Week 9)
To understand which satellite measurements drive the flood detection, we applied permutation importance: each feature is randomly shuffled and the resulting drop in model accuracy is measured. A larger drop indicates a more important feature. This makes the model's decisions transparent and interpretable.

---

## 🗂️ Repository Structure

```
GEOL0069-Valencia-Flood-Detection/
│
├── Valencia_Flood_Detection_GEOL0069.ipynb   # Main Colab notebook
├── README.md                                  # This file
└── figures/
    ├── kmeans_results.png                     # K-means clustering figure
    ├── random_forest_results.png              # Random Forest confusion matrix
    ├── xai_feature_importance.png             # XAI feature importance figure
    └── flood_map_summary.png                  # Before/after/flood map summary
```

---

## 🚀 How to Run

### Requirements
- Google account (free)
- Google Earth Engine account (free) — register at [earthengine.google.com](https://earthengine.google.com)
- Google Colab (free) — no local installation needed

### Steps

1. **Open the notebook in Google Colab:**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - File → Upload notebook → select `Valencia_Flood_Detection_GEOL0069.ipynb`

2. **Authenticate Google Earth Engine (Section 2):**
   ```python
   from google.colab import auth
   auth.authenticate_user()
   ee.Initialize(project='your-project-id')
   ```
   Replace `your-project-id` with your own GEE project ID from [console.cloud.google.com](https://console.cloud.google.com)

3. **Run all cells:**
   - Runtime → Run all
   - Total runtime: approximately 5–10 minutes

### Python Libraries Used
```
earthengine-api    # Google Earth Engine access
geemap             # Interactive map visualisation
numpy              # Numerical computing
matplotlib         # Plotting and figures
scikit-learn       # K-means, Random Forest, XAI
codecarbon         # Environmental cost tracking
Pillow             # Image processing
```

---

## 📊 Results

### K-means Clustering
K-means successfully separated pixels into three physically meaningful clusters. The water/flood cluster showed significantly lower post-flood VV backscatter (mean ≈ −20 to −25 dB) compared to vegetation and urban clusters, consistent with SAR physics.

### Random Forest Classification
The Random Forest classifier achieved **100% accuracy** on the test set. This high accuracy reflects the clean separability of SAR signatures between water and non-water surfaces. The flood map correctly identified the Albufera lagoon and surrounding coastal flood plains as the primary inundated areas.

### Explainable AI
Permutation importance analysis revealed that **pre-flood VH backscatter** is the most important feature for classification, followed by pre-flood VV and the change in VV (diff_VV). This is physically meaningful — knowing the pre-flood baseline is essential for correctly identifying change.

---

## 🌱 Environmental Cost Assessment

The computational cost of this project was measured using [CodeCarbon](https://codecarbon.io/).

| Metric | Value |
|---|---|
| CO₂ emissions (ML computation) | < 1 gCO₂eq |
| Compute platform | Google Colab (cloud) |
| Data access method | Google Earth Engine (server-side) |

**Why this project has a very low carbon footprint:**

Using Google Earth Engine means satellite data is processed on Google's servers rather than downloaded locally. This avoids transferring ~500MB of raw SAR data per image, significantly reducing both energy consumption and storage requirements. Google Colab's shared cloud infrastructure is also more energy-efficient than running equivalent computations on a personal laptop.

For context:
- Sending one email ≈ 4 gCO₂eq
- Boiling a kettle ≈ 15 gCO₂eq
- This entire notebook < 1 gCO₂eq ✅

---

## 📚 References & Data Sources

- **Sentinel-1 SAR data**: European Space Agency (ESA) Copernicus Programme, accessed via [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)
- **Valencia DANA flood event**: Copernicus Emergency Management Service (CEMS) Rapid Mapping Activation EMSR764
- **Course materials**: GEOL0069 AI for Earth Observation, UCL — [cpomucl.github.io/GEOL0069-AI4EO](https://cpomucl.github.io/GEOL0069-AI4EO)
- **K-means & Random Forest**: scikit-learn documentation — [scikit-learn.org](https://scikit-learn.org)
- **CodeCarbon**: Lottick et al. (2019), Energy Usage Reports for Neural Network Training

---

## 👤 Author

**EmmeLai**
UCL Department of Earth Sciences
GEOL0069 AI for Earth Observation
Module Instructor: Dr Michel Tsamados
