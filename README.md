# Project Structure and Pipeline Overview

This repository contains the full data analysis and modeling pipeline for vascular and immune cell analysis.  
The pipeline is modular, with clearly separated stages for **data management**, **feature extraction**, **exploration**, **modeling**, **visualization**, and **result interpretation**.

---

## 📁 Directory Overview

### `data/`
Contains all data files used or produced throughout the project.  
This includes:
- **Raw data**: input CSV files, segmented graphs, or imaging-derived data.
- **Processed data**: files generated after feature extraction, cleaning, or integration.
- **Extracted data**: results of sub-analyses or exported subsets for modeling.


---

### `feature_extraction/`
Includes all scripts related to **computing quantitative features** from data.  
Organized by biological context:

- **`immune_features/`** – Scripts for extracting immune-related descriptors (e.g., T-cell densities, cluster metrics).
- **`vasculature_features/`** – Scripts for extracting vascular graph properties (e.g., vessel density, radius distribution, branching complexity).

Each script outputs structured CSV or graph files to the `data/` directory.


---

### `exploration/`
Used for **data exploration and quality assessment**.  
Contains:
- **Python scripts** and **notebooks** for initial data inspection.
- **`artifacts_exploration/`** – Focused analysis of segmentation or imaging artifacts.

This stage supports exploratory data analysis (EDA), plotting distributions, and identifying preprocessing issues.


---

### `models/`
Houses all modeling scripts.  
Two main subdirectories separate classical ML approaches and graph-based deep learning methods:

- **`classic_models/`** – Scripts for regression, classification, and clustering using tabular features.
- **`gnns/`** – Graph Neural Network architectures and training pipelines for predicting outcomes from vascular/immune graph data.


---

### `output/`
Stores **generated outputs** from any stage of the pipeline.  
Includes:
- Plots, figures, and 3D renderings
- Predictions and tables
- Logs and summary files  
(No Python code should be placed here.)


---

### `results_analysis/`
Contains scripts and notebooks for **interpreting model outputs** and **creating final results**.  
Used for:
- Generating publication-ready plots
- Comparing model performances
- Understanding feature importance and biological interpretation


---

### `napari_visualization/`
Dedicated to **3D visualization** of raw and processed data using Napari.  
Includes:
- Scripts and notebooks for loading segmented volumes or graphs
- Examples demonstrating immune–vascular relationships
- Visual summaries for presentations or publications







