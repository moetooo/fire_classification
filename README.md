# AI/ML Internship Project – Fire Type Recognition and Deforestation Detection (Edunet Foundation)

### Mentor: *Raghunandan M S*  
*Role: Data Analyst & AI Trainer*

---

## Project Title: Identifying Fire Categories Across India (2021–2023) Using MODIS Satellite Data

Each year, India reports numerous fire events ranging from forest and crop fires to other heat anomalies. Being able to distinguish these types using remote sensing data is key for rapid emergency response, ecological assessment, and strategic planning.

NASA’s MODIS sensors on the Terra and Aqua satellites are capable of consistently detecting such thermal events on a global scale. However, developing a model to differentiate between types — like vegetation fires, volcanic heat, static land sources, and offshore anomalies — remains a complex task.

---

## Goal of the Project

Build a machine learning classifier that can distinguish among various fire sources based on MODIS satellite data collected across India between 2021 and 2023.

---

## Dataset Overview

**Source**: NASA FIRMS (Fire Information for Resource Management System)  
**Scope**: India-specific fire detection data (Jan 2021 to Dec 2023)  
**Sensor**: MODIS (Moderate Resolution Imaging Spectroradiometer)  
**Satellites**:  
- **Terra (AM)**: Captures during morning  
- **Aqua (PM)**: Captures during afternoon  

Each satellite provides multiple daily thermal observations for mid-latitude regions such as India.

---

## Detection Methodology

- MODIS identifies thermal activity by analyzing mid-infrared bands (21/22 for fire, 31 for surface temperature).
- Every pixel is tagged with a status: Fire, Non-fire, Water, Cloud, Missing, or Unknown.
- Classification is done using a contextual approach that evaluates pixel surroundings.

> Note: Aqua satellite data may show location imprecision due to real-time orbit calculations. This could result in positional errors of several kilometers in some cases.

---

## Tools & Technologies

- **Programming**: Python 3.12  
- **Workspace**: Jupyter Notebooks (VS Code)  
- **Libraries**:  
  - Data Processing: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - ML Models: `scikit-learn`, `xgboost`  

---

## Applications of This Study

- Monitoring wildfire hotspots  
- Tracking seasonal agricultural burning  
- Predicting environmental impacts  
- Assisting in fire-prone zone mapping  
- Mapping deforestation patterns over time

---

## Useful Resources

- [NASA FIRMS: About & Documentation](https://www.earthdata.nasa.gov/data/tools/firms)  
- [Fire Data Downloads](https://firms.modaps.eosdis.nasa.gov/download/)

---

# Progress Tracker

## Week 1 – Completed

**Highlights:**

- Collected and loaded MODIS fire datasets for the years 2021, 2022, and 2023.
- Merged all yearly datasets into a single DataFrame for unified processing.
- Performed comprehensive data cleaning:
  - Verified dataset dimensions, column names, and data types.
  - Checked for missing values and duplicates.
  - Reviewed descriptive statistics of numerical features.
- Conducted exploratory data analysis (EDA):
  - Analyzed fire type distribution using count plots.
  - Inspected frequency of each categorical value.
  - Visualized the distribution of the `confidence` feature, identifying bimodal patterns.
- Noted an imbalance in fire type labels, which may impact model performance in later stages.


