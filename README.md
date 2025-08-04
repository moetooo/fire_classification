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

## Week 2 – Completed 

**Highlights:**

- Created new time-based features from the timestamp such as:
  - `Month` of detection  
  - `Day of the week`  
  - `Hour` of the day  
- Visualized seasonal and hourly trends in fire activity across regions.
- Detected and removed extreme values using the **Interquartile Range (IQR)** method.
- Applied **One-Hot Encoding** to handle categorical variables effectively.
- Standardized numerical columns using `StandardScaler` for uniform feature scaling.
- Addressed data imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique) to improve model generalization.

---

## Week 3 – Completed

**Highlights:**

- Split the dataset into training and test sets (e.g., 80-20 split).
- Trained and tested several classification models:
  - **Logistic Regression**
  - **K-Nearest Neighbors (KNN)**
  - **Decision Tree**
  - **Random Forest**
- Evaluated each model using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
- Finalized **Random Forest** as the best-performing model with ~98% accuracy.
- Exported the trained model and preprocessor using `joblib` for deployment.
- Built an interactive prediction app using **Streamlit** to input fire data and display predicted fire categories in real-time.

## 🛠️ Tools & Technologies

- **Language**: Python 3.11  
- **Modeling**: Scikit-learn (Random Forest Classifier)  
- **Interface**: Streamlit  
- **Libraries**: `pandas`, `numpy`, `joblib`, `pickle`  
- **IDE**: Jupyter Notebook, VS Code  

---

## 📁 Files Included

- `app.py` – Streamlit web app  
- `fire_type_model.pkl` – Trained classification model  
- `label_encoders.pkl` – Encoded objects for categorical fields  
- `requirements.txt` – Python dependencies  

---

## 📦 Installation

1. **Clone the repository**  
```bash
git clone https://github.com/moetooo/fire-type-prediction.git
cd fire-type-prediction