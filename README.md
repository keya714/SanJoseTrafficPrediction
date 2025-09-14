# Traffic Accident High-Risk Zone Prediction

This project utilizes PySpark, Federated learning and TensorFlow to build a predictive model for identifying high-risk motor vehicle accident zones based on demographic and location data. The system processes large datasets efficiently and applies a neural network model to predict accident hotspots, supporting targeted public safety interventions.

---

## Features

- **Big Data Processing with PySpark:**  
  Efficiently loads, cleans, and preprocesses a large traffic fatality dataset using scalable distributed computing.

- **Binary Label for High-Risk Zones:**  
  Identifies top 10 zip codes with the highest accident counts and labels incidents accordingly for supervised learning.

- **Feature Engineering:**  
  Encodes categorical variables (`Gender`, `Incident Zip`) and scales numerical features (`Age`) to prepare for model training.

- **Neural Network Model with TensorFlow:**  
  Implements a feed-forward neural network trained to distinguish high-risk zones using demographic and location features.

- **Evaluation Metrics:**  
  Uses AUC (Area Under ROC Curve) to measure model performance, ensuring reliable classification quality.

---

## Getting Started

### Prerequisites

- Python 3.x
- PySpark
- TensorFlow
- scikit-learn
- Pandas
- NumPy

Install Python dependencies using pip:

pip install pyspark tensorflow scikit-learn pandas numpy

### Dataset

The dataset "Medical Examiner-Coroner Motor Vehicle Deaths" should be available in CSV format. Update the file path in the code to point to your local copy:

data = spark.read.csv(
r"C:\path\to\Medical_Examiner-Coroner,_Motor_Vehicle_Deaths_dataset.csv",
header=True,
inferSchema=True
)
---

## How It Works

1. **Data Loading and Cleaning:**  
   The dataset is loaded into a PySpark DataFrame, dropping rows missing critical columns (`Age`, `Gender`, `Incident Zip`).

2. **Label Creation:**  
   A `HighRisk` binary label is created based on whether the incident occurred in one of the top 10 accident-prone zip codes.

3. **Feature Engineering:**  
   Categorical features are indexed using PySpark's `StringIndexer`. Numeric features are scaled in Pandas.

4. **Model Training:**  
   The processed data is converted to a Pandas DataFrame and used to train a TensorFlow neural network.

5. **Evaluation:**  
   Model performance is evaluated on a test set, reporting AUC to measure discrimination between high-risk and other zones.

---

## Federated Learning Potential

While the current implementation trains on centralized data, the architecture facilitates extension to federated learning. This would allow multiple clients (e.g., municipalities or hospitals) to train local models on private data and collaboratively improve a global model without sharing sensitive raw data, addressing privacy and regulatory concerns.

---

## Benefits and Use Cases

- Targeting traffic safety interventions in high-risk areas.
- Informing infrastructure planning and emergency response.
- Enhancing insurance risk modeling.
- Supporting collaborative and privacy-preserving modeling in distributed environments.

---

## Running the Project

Execute the main Python script after adjusting dataset paths and environment setups. The training logs will show model metrics, and final AUC will indicate predictive accuracy.

---


## Acknowledgments

This project is inspired by public health and safety initiatives leveraging big data anal
