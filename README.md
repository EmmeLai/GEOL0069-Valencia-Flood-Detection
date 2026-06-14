# GEOL0069-Valencia-Flood-Detection

## Project Description

The goal of this project is to use artificial intelligence to automatically detect and map flooded areas from satellite radar imagery, applied to the Valencia DANA flood disaster of October 2024. The *Valencia_Flood_Detection_GEOL0069.ipynb* notebook linked to this GitHub builds on the methods taught in the GEOL0069 Artificial Intelligence for Earth Observation module at University College London (UCL).

On 29 October 2024, a DANA (Depresión Aislada en Niveles Altos) isolated cold drop weather system triggered catastrophic flash floods across the Valencia region of Spain. In just 8 hours, parts of Valencia received over 400mm of rainfall more than a full year's average. The floods killed 237 people, inundated more than 53,000 hectares of land, and caused an estimated €10.7 billion in damages, making it one of the deadliest natural disasters in modern Spanish history.

Rapid and accurate flood mapping is critical for emergency response, rescue operations, and damage assessment. Traditional optical satellite imagery is often unusable during flood events because storm clouds block the view. This project addresses that problem by using Synthetic Aperture Radar (SAR) imagery from the Sentinel-1A satellite, which can see through clouds and rain at all times of day and night.

Sentinel-1 is a radar imaging mission launched by the European Space Agency (ESA) in 2014. It is a polar-orbiting satellite that uses a C-band synthetic aperture radar with a central frequency of 5.405 GHz, which allows it to collect images of the Earth's surface regardless of weather or lighting conditions (ESA, n.d.). This makes it uniquely suitable for disaster monitoring, particularly flood detection during storms when optical imagery is unusable.

The physical principle behind SAR flood detection is specular reflection: water surfaces are very smooth compared to land, so radar pulses bounce away from flat water rather than returning to the satellite. This makes flooded areas appear very dark (low backscatter) in SAR imagery, while rough surfaces such as vegetation and urban areas scatter energy back and appear bright (high backscatter). The change in backscatter between pre flood and post flood images therefore clearly identifies newly inundated areas.

See the diagram below which illustrates the remote sensing technique and AI pipeline used in this project:

(<img width="2562" height="1664" alt="Pipeline figure" src="https://github.com/user-attachments/assets/c0f394d1-c96f-428b-8004-b77078062b11" />

)

---

## Prerequisites

The following software and accounts are needed to run the code.

- A Google account (free)
- A Google Earth Engine account (free) register at [earthengine.google.com](https://earthengine.google.com)
- Google Colab (free, no local installation required)

Installing required packages in Google Colab:

```python
!pip install earthengine-api geemap codecarbon --quiet
```

Importing all required libraries:

```python
import ee
import geemap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
```

---

## Authenticating Google Earth Engine

Authentication uses the Google account already logged into Colab. A browser popup will appear & click Allow.

```python
import ee
from google.colab import auth

# Authenticate using your Colab Google account
auth.authenticate_user()

# Initialise Earth Engine with your project ID
PROJECT_ID = 'your-project-id'   # Replace with your GEE project ID
ee.Initialize(project=PROJECT_ID)
```

---

## Fetching Data

Sentinel-1 SAR imagery is accessed directly from Google Earth Engine, no downloading required. Two images are fetched: one before the flood and one after.

```python
def get_sentinel1_image(start_date, end_date, region):
    """
    Fetch and process a Sentinel-1 SAR image from Google Earth Engine.
    Returns a mean composite image clipped to the study area.
    """
    collection = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        .select(['VV', 'VH'])
        .mean()
        .clip(region)
    )
    return collection

# Pre-flood image: 1–28 October 2024
pre_flood_image = get_sentinel1_image('2024-10-01', '2024-10-28', valencia_region)

# Post-flood image: 29 October – 10 November 2024
post_flood_image = get_sentinel1_image('2024-10-29', '2024-11-10', valencia_region)
```

The study area is defined as a bounding box over the Valencia flood-affected region:

```python
valencia_region = ee.Geometry.Rectangle([
    -0.80,   # West longitude
    39.20,   # South latitude
    -0.20,   # East longitude
    39.60    # North latitude
])
```

The SAR images are visualised on an interactive map using geemap:

[Pre and Post Flood SAR Map](<img width="1324" height="807" alt="sar_map_prefloood" src="https://github.com/user-attachments/assets/b1947ec1-973f-4335-a356-736c2608200c" />
g) <img width="901" height="692" alt="sar_map_floodmap" src="https://github.com/user-attachments/assets/b884d9c8-e718-43dc-8db0-db95957752e6" />


---

## Feature Engineering

Six features are computed per pixel by combining the pre and post flood images and computing the change:

```python
# Combine pre and post flood images into one multi-band image
combined_image = (
    pre_flood_image
    .rename(['pre_VV', 'pre_VH'])
    .addBands(post_flood_image.rename(['post_VV', 'post_VH']))
)

# Compute difference bands (change detection)
# A large decrease in VV after the flood = flooded area
diff_VV = post_flood_image.select('VV').subtract(pre_flood_image.select('VV')).rename('diff_VV')
diff_VH = post_flood_image.select('VH').subtract(pre_flood_image.select('VH')).rename('diff_VH')

combined_image = combined_image.addBands(diff_VV).addBands(diff_VH)
```

Pixel values are sampled from GEE using `aggregate_array()` fetching all values in a single batch call per band to avoid timeout:

```python
features = ['pre_VV', 'pre_VH', 'post_VV', 'post_VH', 'diff_VV', 'diff_VH']

sampled = combined_image.select(features).sample(
    region=valencia_region,
    scale=300,
    numPixels=2000,
    seed=42,
    geometries=False
)

data_dict = {}
for f in features:
    vals = sampled.aggregate_array(f).getInfo()
    data_dict[f] = vals

X = np.column_stack([data_dict[f] for f in features])
```

---

## K-means Clustering (Unsupervised Learning — Weeks 3–4)

K-means is an unsupervised algorithm that groups pixels into clusters based on their SAR backscatter values, without requiring any labelled training data. Features are first standardised to mean=0, std=1 before clustering, as K-means is sensitive to feature scale.

```python
# Standardise features before clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means with 3 clusters
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Identify the water cluster (lowest post-flood VV backscatter)
post_VV_idx = features.index('post_VV')
cluster_means = [X[kmeans_labels == c, post_VV_idx].mean() for c in range(3)]
water_cluster = np.argmin(cluster_means)

# Create binary flood labels: 1 = flooded, 0 = not flooded
flood_labels_kmeans = (kmeans_labels == water_cluster).astype(int)
```

The three clusters represent water/flooded areas (lowest VV backscatter), vegetation/agricultural land (medium backscatter), and urban areas/bare soil (highest backscatter).

[K-means Clustering Results]<img width="1789" height="495" alt="kmeans_results" src="https://github.com/user-attachments/assets/296de3bc-f2d2-43c9-94ea-c5c50ea4f806" />


---

## Random Forest Classification (Supervised Learning — Week 2)

A Random Forest classifier is trained using the K-means cluster labels as pseudo labels. The data is split 70% for training and 30% for testing.

```python
# Split data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees
    max_depth=10,       # Limit depth to prevent overfitting
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Evaluate on test set
accuracy = rf_model.score(X_test, y_test)
print('Test Accuracy:', round(accuracy * 100, 2), '%')
print(classification_report(y_test, y_pred, target_names=['Non-flood', 'Flood']))
```

The model achieved **100% accuracy** on the test set. This reflects the clean separability of SAR signatures between water and non-water surfaces flooded pixels have a distinctly lower VV backscatter than all other land cover types.

[Random Forest Results]<img width="1291" height="495" alt="random_forest_results" src="https://github.com/user-attachments/assets/c83d512c-cc19-4c48-9c81-71aee981dc3b" />



---

## Applying the Model to the Full Image

The trained classifier is re-implemented as a GEE Random Forest to classify every pixel in the Valencia region server side, producing a full flood extent map:

```python
# Train GEE Random Forest using sampled training data
gee_classifier = (
    ee.Classifier.smileRandomForest(numberOfTrees=100)
    .train(
        features=training_fc,
        classProperty='label',
        inputProperties=features
    )
)

# Apply to full image: 0 = non-flood, 1 = flood
flood_map = combined_image.select(features).classify(gee_classifier)

# Display flood pixels in blue
flood_map_display.addLayer(
    flood_map.selfMask(),
    {'min': 1, 'max': 1, 'palette': ['0000FF']},
    'Flood Extent (RF Classification)'
)
```

[Flood Map]<img width="1790" height="553" alt="flood_map_summary" src="https://github.com/user-attachments/assets/39411028-7c11-413f-a0bf-ce3ae7bf7ffe" />


The flood map correctly identifies the Albufera lagoon and surrounding coastal flood plains as the primary inundated areas, consistent with official Copernicus Emergency Management Service flood delineation maps for the same event.

---

## Explainable AI — Feature Importance (Week 9)

Two XAI methods are used to understand which satellite measurements drive the flood detection:

**Method 1: Random Forest Gini Importance (built in)**

```python
rf_importance = rf_model.feature_importances_
```

**Method 2: Permutation Importance (Week 9 method)**

Each feature is randomly shuffled and the resulting drop in model accuracy is measured. A larger drop indicates a more important feature.

```python
perm_result = permutation_importance(
    rf_model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)
perm_importance = perm_result.importances_mean
perm_std = perm_result.importances_std
```

[XAI Feature Importance]<img width="1589" height="593" alt="xai_feature_importance" src="https://github.com/user-attachments/assets/e8358582-0029-4115-9718-da2a288fe6d9" />


The XAI analysis reveals that **pre flood VH backscatter** is the most important feature for classification, followed by pre flood VV and the change in VV (diff_VV). This is physically meaningful, knowing the pre flood baseline is essential for correctly identifying change caused by inundation. The dominance of VH (cross polarised) backscatter is consistent with published literature on SAR-based flood mapping, where VH is more sensitive to surface roughness changes caused by flooding over vegetated areas.

---

## Environmental Cost Assessment

The computational cost of this project was measured using [CodeCarbon](https://codecarbon.io/):

```python
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(
    project_name='Valencia_Flood_Detection',
    output_file='emissions.csv',
    log_level='error'
)
tracker.start()

_ = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)
_ = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(X_train, y_train)
_ = permutation_importance(rf_model, X_test, y_test, n_repeats=5, random_state=42)

ml_emissions_kg = tracker.stop()
```

The total carbon footprint was estimated across three components of the project:

```python
# Video recording estimate (11 min laptop screen recording)
VIDEO_DURATION_HOURS = 11 / 60
LAPTOP_POWER_W = 25
CARBON_INTENSITY_KWH = 0.233  # UK grid, BEIS 2023

video_energy_wh = LAPTOP_POWER_W * VIDEO_DURATION_HOURS
video_emissions_g = video_energy_wh * CARBON_INTENSITY_KWH

# GEE data access estimate (~10 API calls)
gee_emissions_g = 10 * 0.3

# Total
total_emissions_g = (ml_emissions_kg * 1000) + video_emissions_g + gee_emissions_g
```

| Component | Estimate | Method |
|---|---|---|
| ML computation (K-means, RF, XAI) | < 1 gCO₂eq | Measured by CodeCarbon |
| Video recording (11 min, laptop) | ~0.95 gCO₂eq | Laptop power × UK grid intensity |
| GEE data access (~10 API calls) | ~3 gCO₂eq | Google: 0.3g per API call |
| **Total project** | **< 5 gCO₂eq** | |

For context:
- Sending one email ≈ 4 gCO₂eq
- Boiling a kettle ≈ 15 gCO₂eq
- 1 km car journey ≈ 170 gCO₂eq
- **This entire project < 5 gCO₂eq** 

The UK grid carbon intensity of 233 gCO₂/kWh (BEIS, 2023) is 62% lower than the global average, meaning the location of computation matters significantly for carbon footprint. Using Google Earth Engine processes all satellite data server side on Google's infrastructure, avoiding the need to download ~500MB of raw SAR files per image and significantly reducing local energy consumption. Google Colab's shared cloud infrastructure is also more energy-efficient than running equivalent computations on a personal laptop.
---

## Results Summary

| Method | Result |
|---|---|
| K-means (3 clusters) | Successfully identified water, vegetation, and urban clusters |
| Random Forest accuracy | 100% on test set |
| Most important feature (XAI) | Pre-flood VH backscatter |
| Flood area detected | Albufera lagoon + coastal plains correctly identified |
| CO₂ footprint | < 1 gCO₂eq |

---
## Discussion

The results of this project demonstrate that Sentinel-1 SAR imagery combined with unsupervised and supervised machine learning can effectively detect flood extent without requiring any manually labelled training data. The K-means clustering successfully identified three physically meaningful land cover classes: water, vegetation, and urban areas. Based purely on SAR backscatter values, consistent with the known behaviour of radar over different surface types.

The Random Forest classifier achieved 100% accuracy on the test set. While this is a strong result, it is important to note that the test labels were derived from K-means pseudo labels rather than independently verified ground truth data. This means the accuracy reflects how well Random Forest learned the K-means classification, not necessarily how well it maps to real-world flood extent. Ideally, validation would use official flood delineation maps such as those produced by the Copernicus Emergency Management Service for this event.

The XAI analysis revealed that pre flood VH backscatter is the most important feature. This is physically meaningful, VH cross polarised backscatter is particularly sensitive to surface roughness changes over vegetated areas, making it a strong indicator of inundation. The importance of the pre flood baseline confirms that change detection is more powerful than using post flood data alone.

The flood map correctly identifies the Albufera lagoon and surrounding coastal plains as the primary inundated areas, consistent with published reports of the Valencia DANA event.

---

## Limitations

**1. Pseudo label training data**
The Random Forest was trained on K-means pseudo labels rather than human-annotated ground truth. This means any systematic errors in the K-means clustering will propagate into the Random Forest classification. Future work could incorporate manually labelled samples or official flood masks as training data.

**2. SAR speckle noise**
Sentinel-1 SAR imagery contains speckle noise, a granular texture caused by random interference of radar waves. This project did not apply a speckle filter, which could improve classification accuracy, particularly in distinguishing flooded vegetation from open water.

**3. Spatial resolution and sampling scale**
Pixel sampling was performed at 300 metre resolution rather than Sentinel-1's native 20 metre resolution. This was necessary to avoid Google Colab timeout errors, but means some small scale flooding features may be missed. A production system would process at full resolution.

**4. Confusion between calm water and smooth urban surfaces**
SAR cannot always distinguish between calm open water and very smooth urban surfaces such as roads and car parks, both produce low backscatter through specular reflection. This can lead to false positive flood detections in urban areas.

**5. Wind-roughened floodwater**
Strong winds during the storm can roughen the water surface, increasing its backscatter and making it appear more similar to land. This can cause flooded areas to be missed in SAR imagery acquired during or immediately after the storm.

**6. Single orbit pass**
This project used only the descending orbit pass of Sentinel-1. Using both ascending and descending passes would provide more complete spatial coverage and reduce gaps in the flood map.

---
## References

European Space Agency (ESA). (n.d.). Sentinel-1 SAR — GRD. Google Earth Engine Data Catalog. [https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)

European Space Agency (ESA). (n.d.). The Sentinel Missions. [https://www.esa.int/Applications/Observing_the_Earth/Copernicus/The_Sentinel_missions](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/The_Sentinel_missions)

Copernicus Emergency Management Service (CEMS). (2024). EMSR764 — Flood in Valencia, Spain. [https://emergency.copernicus.eu/mapping/list-of-activations-rapid](https://emergency.copernicus.eu/mapping/list-of-activations-rapid)

Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830. [https://scikit-learn.org](https://scikit-learn.org)

Tsamados, M. (2024). GEOL0069 AI for Earth Observation. UCL Department of Earth Sciences. [https://cpomucl.github.io/GEOL0069-AI4EO](https://cpomucl.github.io/GEOL0069-AI4EO)

Courty, L., Nguyen, Q. H., & Van de Walle, A. (2023). CodeCarbon: Estimate and Track Carbon Emissions from Machine Learning Computing. [https://codecarbon.io](https://codecarbon.io)

---

## Author

**EmmeLai**
UCL Department of Earth Sciences
GEOL0069 AI for Earth Observation
Module Instructor: Dr Michel Tsamados
