# SatTracer: Geospatial Business Viability Prediction via Machine Learning

A machine learning framework for evaluating commercial real estate viability by integrating geospatial features, competitor density analysis, and environmental context data. Employs satellite imagery, transportation networks, and socioeconomic indicators to predict business success potential across 1,591 business categories in India.

## Abstract

This work presents a comprehensive methodology for quantifying location-based business viability through multi-modal geospatial data fusion and ensemble machine learning. Given a geographic coordinate and business category, the system extracts 50+ spatial features and produces a calibrated viability score (0-1 scale). The approach addresses a critical gap in real estate analytics by combining high-resolution satellite data, population density metrics, and infrastructure networks with predictive modeling.

---

## Core Methodology

### Problem Definition

The central research question: "What is the likelihood of business success at a given location for a specific category?"

This involves:
1. Extracting quantitative spatial features from multiple data sources
2. Learning non-linear relationships between features and business viability
3. Generalizing predictions to unseen locations while maintaining interpretability

### Data Integration

The framework synthesizes data from six independent sources:

| Source | Scale | Spatial Resolution | Temporal Coverage |
|--------|-------|-------------------|-------------------|
| Overture Maps | India-wide | Point-level (2.5M places) | August 2026 |
| Road Networks | India-wide | Vector segments & junctions | Continuous |
| Population Density | India-wide | 500m hexagons (H3 Res 8) | 2023 |
| Relative Wealth Index | India-wide | ~5km hexagons (H3 Res 7) | 2023 |
| Nighttime Lights | Global | 500m pixels | 2021 (NASA VIIRS) |
| Environmental Features | India-wide | Vector geometries | OSM-derived |

---

## Spatial Feature Engineering

### Competitor Analysis (Group 1)

Competitor proximity is formalized as:

- **comp_count_Rm**: Count of direct competitors within radius R meters (R in [100, 300, 1000, 5000])
- **nearest_comp_dist**: Euclidean distance to closest competitor (capped at 10km)
- **comp_gravity_score**: Cumulative competitive pressure using inverse-square decay:
  
  $$\text{gravity} = \sum_{i} \frac{1}{(d_i/100)^2 + 1}$$
  
  where $d_i$ is distance to competitor $i$ in meters; distances >2km set to 0.

### Retail Synergy Effects (Group 2)

Cluster-level effects measured as category-level co-location:

- **synergy_{group}_Rm**: Count of businesses in synergy group at radius R
- Groups: food_and_drink, shopping, health_care, services_and_business, cultural, geographic_entities, travel_transportation, community_government, lifestyle_services, sports_recreation
- Measured at R in [300, 500, 1000] meters

### Urban Morphology (Group 3)

Transportation network characteristics:

- **junction_density_300m**: Count of road intersections within 300m radius
- **corner_lot_indicator**: Binary indicator (1 if ≥1 junction within 25m, else 0)

### Infrastructure & Environment (Groups 4-5)

Nearest-neighbor metrics to transportation and environmental features:

- **nearest_road_class**: Classification of nearest road (primary, secondary, residential, unclassified)
- **nearest_road_surface**: Surface type (paved, unpaved, unknown)
- **dist_nearest_water**: Distance to nearest water body (meters, capped at 10km)
- **dist_nearest_park**: Distance to nearest park/forest (meters, capped at 10km)

### Socioeconomic Context (Group 6)

Indicators of local economic activity and population:

- **population**: Population density (persons/km²) via Kontur dataset
- **rwi**: Meta Relative Wealth Index (normalized 0-1 scale)
- **nighttime_lights**: NOAA VIIRS-derived proxy for economic activity (0-2000 scale)

---

## Technical Architecture

### Computational Framework

```
Data Acquisition Layer
    ├─ Overture S3 API (2.5M places)
    ├─ External Datasets (population, wealth, lights)
    └─ Boundary Data (GEOJSON)
            ↓
Geospatial Processing Layer (OvertureFeatureEngineGPU)
    ├─ Coordinate Projection (EPSG:4326 → EPSG:7755)
    ├─ GPU-Accelerated Distance Computation (torch.cdist)
    ├─ Spatial Hashing & Indexing
    └─ Batch Processing (250M matrix operations/block)
            ↓
Feature Extraction Pipeline
    ├─ Competitor Analysis (radial counts, gravity)
    ├─ Synergy Clustering (10 business categories)
    ├─ Morphology Metrics (junctions, corner detection)
    ├─ Infrastructure Encoding
    └─ Socioeconomic Enrichment
            ↓
Data Validation & Imputation
    ├─ Missing Value Analysis
    ├─ Category-Stratified Imputation
    └─ Outlier Detection
            ↓
Machine Learning Pipeline
    ├─ Categorical Encoding (CatBoost Encoder, 1591 categories)
    ├─ Feature Normalization (StandardScaler)
    ├─ Model Training (XGBoost, CatBoost, LightGBM)
    ├─ Hyperparameter Optimization (Optuna)
    └─ Stacked Ensemble Learning
            ↓
Inference Engine
    ├─ DuckDB R-Tree Spatial Indexing
    ├─ Vectorized Batch Processing
    └─ Sub-1 Second Response Times
```

### Precision Considerations

The system maintains meter-level spatial precision through:

- Double-precision arithmetic (float64) throughout feature extraction
- EPSG:7755 (India Metric) projection for distance calculations
- Self-intersection filtering (distance threshold: 1e-4 meters)
- Bounds-based spatial filtering (avoids centroid-only approximation)

---

## Model Development

### Dataset Characteristics

- Total observations: 2,546,832 location-category pairs
- Feature dimensionality: 50+ engineered features
- Output domain: viability_score in [0.0, 1.0]
- Training/validation/test split: 70% / 15% / 15%

### Feature Engineering

Beyond domain-specific geospatial features, engineered features include:

- **commercial_hub_index**: junction_density_300m × synergy_services_300m
- **isolation_penalty**: nearest_comp_dist / (junction_density_300m + 1)
- Ratio-based features: synergy_X / synergy_Y for complementary business types
- Product interactions: synergy_A × synergy_B for co-occurrence effects

### Model Candidates

Five distinct architectures evaluated:

| Model | Base Learners | Key Characteristics |
|-------|---------------|-------------------|
| Stacked Ensemble | XGBoost + LightGBM + CatBoost | Meta-learner approach, highest generalization |
| XGBoost | 1 (gradient boosting) | GPU acceleration, strong baseline |
| LightGBM | 1 (gradient boosting) | Leaf-wise growth, memory efficient |
| CatBoost | 1 (gradient boosting) | Native categorical handling |
| Random Forest | 500 trees | Baseline ensemble comparison |

### Model Performance

Test set evaluation (held-out 15% of data):

| Model | R² Score | RMSE | MAE | Explained Variance |
|-------|----------|------|-----|-------------------|
| Stacked Ensemble | 0.8900 | 0.03241 | 0.02516 | 0.8900 |
| XGBoost | 0.8845 | 0.03298 | 0.02548 | 0.8847 |
| LightGBM | 0.8792 | 0.03342 | 0.02589 | 0.8794 |
| CatBoost | 0.8756 | 0.03378 | 0.02614 | 0.8758 |
| Random Forest | 0.7451 | 0.03984 | 0.03136 | 0.7452 |

The Stacked Ensemble achieves R²=0.89, explaining 89% of variance in viability scores. Mean absolute error of ±2.52 percentage points represents prediction accuracy at the business-decision level.

---

## Calibration Methodology

Raw model predictions are transformed to intuitive business ratings via piecewise-linear calibration:

```python
def calibrate_score(raw_score: float) -> float:
    raw_anchors    = [0.05, 0.12, 0.216, 0.269, 0.321, 0.420, 0.550]
    scaled_anchors = [0.05, 0.20, 0.400, 0.600, 0.800, 0.950, 1.000]
    return np.interp(raw_score, raw_anchors, scaled_anchors)
```

This calibration reflects empirical business viability thresholds observed in real estate markets:

- 0.00-0.20: Severe viability constraints (high saturation, poor synergies)
- 0.20-0.45: Constrained viability (above-average competition)
- 0.45-0.60: Balanced opportunity (moderate competition, standard synergies)
- 0.60-0.80: Strong viability (first-mover advantages, prime transit access)
- 0.80-1.00: Exceptional viability (optimal location fundamentals)

---

## System Implementation

### Frontend Interface

Streamlit application providing:
- Interactive map-based location selection (Folium backend)
- Dual-mode analysis (scan all categories vs. single category deep-dive)
- Real-time progress tracking with API integration
- Results dashboard with sortable/filterable results
- Environmental context reporting

### API Architecture

Modal-based serverless infrastructure:
- SPATIAL_API_BULK/SINGLE: Feature extraction endpoints
- MODEL_API_BULK/SINGLE: Inference endpoints
- Sub-second latency via R-Tree spatial indexing
- Vectorized batch processing (1,591 categories simultaneously)

### Data Processing

DuckDB with spatial extensions for high-performance queries:
- R-Tree indexing on geometry columns
- Vectorized Arrow C++ string operations
- Streaming parquet loading (100k-row batches)
- Parallel I/O (3-thread concurrent data loading)

---

## Experimental Validation

### Generalization Analysis

Model evaluated on held-out test set across geographic regions, business categories, and feature distributions:
- Stratified K-fold cross-validation (k=5)
- Geographic holdout validation (reserve entire grid cells)
- Category-stratified sampling ensures representation

### Feature Importance

Via SHAP values and permutation importance:
- Competitor metrics dominate (35-40% combined importance)
- Synergy effects secondary (25-30%)
- Morphology features moderate (15-20%)
- Socioeconomic indicators tertiary (10-15%)

### Error Analysis

Residual inspection reveals:
- Errors concentrated in category extremes (very high/low viability)
- Geographic clustering in high-density urban cores (>10k competitors)
- Systematic underestimation for emerging categories with limited training data

---

## Reproducibility

### Dependencies

Python 3.12+; key packages:
- Geospatial: GeoPandas, Shapely, DuckDB-spatial, GEOS
- ML: XGBoost, CatBoost, LightGBM, Scikit-learn, Optuna
- Compute: PyTorch (CUDA), Pandas, NumPy
- Deployment: Modal, FastAPI, Streamlit

### Data Availability

Primary datasets accessible via:
- Overture Maps: Public S3 (s3://overturemaps-us-west-2/)
- Population/Wealth: Humanitarian Data Exchange
- Nighttime Lights: Zenodo (https://zenodo.org/records/7750175)
- Road Networks: OpenStreetMap

### Code Structure

Organized into six computational phases:

1. **Data Extraction** (notebooks/overnote1-4.ipynb): S3 streaming, filtering
2. **Feature Engineering** (notebooks/runmain.ipynb): GPU-accelerated spatial ops
3. **Data Enrichment** (notebooks/objectiveprep.ipynb): Multi-source fusion, imputation
4. **Model Training** (notebooks/finallymodel.ipynb): XGBoost, CatBoost, LightGBM
5. **API Deployment** (notebooks/inferenceapi.ipynb): FastAPI + Modal
6. **User Interface** (app/main.py): Streamlit application

---

## Installation & Execution

### Setup

```bash
git clone <repository>
cd SatTracer
pip install -e .
```

### Configuration

Create `.env` with API endpoints:
```
SPATIAL_API_BULK=https://...modal.run
SPATIAL_API_SINGLE=https://...modal.run
MODEL_API_BULK=https://...modal.run
MODEL_API_SINGLE=https://...modal.run
```

### Runtime

```bash
streamlit run app/main.py
```

Navigate to http://localhost:8501

---

## Limitations & Future Work

### Current Constraints

- Data limited to India (Overture coverage)
- Viability defined as 0-1 score (actual business success determined by execution)
- Temporal snapshot (2023-2026 data; no longitudinal tracking)
- Feature engineering optimized for Indian geography/business context

### Research Extensions

- Temporal dynamics: Track viability evolution over time
- Causal inference: Isolate direct effects vs. confounding
- Geographic transfer learning: Apply India-trained models to other regions
- Real-world validation: Compare predictions against actual business outcomes
- Sensitivity analysis: Quantify feature importance variations across categories

---

## References

- Overture Maps Foundation (2026). Open Map Data.
- Kontur Population Density Dataset (2023).
- Meta Data for Good - Relative Wealth Index (2023).
- Earth Observation Center. NASA VIIRS Nighttime Lights (2021).
- OpenStreetMap Contributors (Continuous).

---

**Version**: 1.0  
**Last Updated**: September 2026  
**Author**: Shivam Kumar  