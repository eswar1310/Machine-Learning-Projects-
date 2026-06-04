# Predictive Machine Learning Portfolio 📊

## Overview
This repository contains a curated collection of machine learning classification models focused on business operations, risk forecasting, and predictive modeling. The portfolio features three primary case studies: French Motor Insurance Claims classification, Credit Card Application Approval scoring, and Amazon Sales Order status forecasting.

## Problem Statement
Industrial systems, financial lenders, and retail fulfillment channels produce large volumes of transactional data. Building models that can predict risk (e.g. insurance claims, credit default, order rejection) requires addressing real-world challenges, such as handling highly imbalanced datasets, performing feature engineering, and optimizing classification thresholds to minimize financial loss.

## Solution
Each project in this portfolio applies a structured machine learning pipeline to address these challenges:
1. **French Motor Insurance claims classification**: Integrates claim frequency and claim severity datasets to classify likelihood of claims.
2. **Credit Card Application Approval scoring**: Trains ensemble models to classify applicant approval risk.
3. **Amazon Sales Order classification**: Visualizes sales data trends and predicts order rejection events.

## Features
* **Exploratory Data Analysis (EDA)**: Thorough distributions, correlation matrices, and imbalance reviews.
* **Comprehensive Preprocessing**: Implements standard scaling, label encoding, and missing value imputation.
* **Imbalanced Class Optimization**: Implements custom thresholding, evaluating classifiers using precision, specificity, sensitivity, and balanced accuracy.
* **Model Benchmarking**: Side-by-side comparison of Logistic Regression, Decision Trees, Random Forests, Extra Trees, Support Vector Machines (SVM), and LightGBM.

## Architecture
The standard machine learning workflow executed across these notebook studies:

```mermaid
graph LR
    A[Raw Datasets / CSVs] -->|Load / Clean| B[Exploratory Data Analysis]
    B -->|Imputation / Scaling| C[Feature Engineering]
    C -->|Split train-test| D[Model Training & Hyperparameter Tuning]
    D -->|Evaluate Models| E[Balanced Metrics Scorecard]
    E -->|Select Best Model| F[Model Persistence / Serialization]
```

## Technology Stack
* **Languages**: Python (Jupyter Notebooks).
* **Libraries**: Scikit-Learn, LightGBM, Pandas, NumPy, SciPy, Matplotlib, Seaborn, Joblib.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/eswar1310/predictive-ml-portfolio.git
   cd predictive-ml-portfolio
   ```
2. Set up virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   ```
3. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure Scikit-learn, LightGBM, Pandas, and Seaborn are installed).*

## Usage
Start Jupyter and open the case studies:
```bash
jupyter notebook
```
* **French Motor Claims**: Run `Motor_Claims/French_Motor_Project.ipynb`.
* **Credit Card Approval**: Run `Credit_Card_Approval/Credit Card Application Approval .ipynb`.
* **Amazon Order Classification**: Run `Amazon Orders Classification/Amazon_sales (1).ipynb`.

## Results
* **French Motor Insurance**: Successfully merged frequency and severity datasets to construct a claim predictor. Ensemble methods (Random Forest and Extra Trees) outperformed baseline classifiers.
* **Credit Card Scoring**: Benchmarked LightGBM and Extra Trees, achieving balanced accuracy by tuning the decision threshold to trade off sensitivity and specificity.
* **Amazon Orders**: Visualized key state and product category distribution channels, implementing classification scoring for order rejections.

## Screenshots
*(Provide links or placeholders to repository social previews)*
* **EDA Correlation Plot**: `[Insert Correlation Plot Image]`
* **Model Comparison Table**: `[Insert Score Table Screenshot]`

## Future Improvements
* Add notebook files testing neural network classifiers (Multi-Layer Perceptrons) in PyTorch.
* Set up MLflow tracking to log hyperparameters and classification metrics across training runs.
* Deploy the credit scoring model as a microservice API using FastAPI.

## Project Structure
```text
predictive-ml-portfolio/
├── Amazon Orders Classification/
│   ├── Amazon Sale Report.csv
│   └── Amazon_sales (1).ipynb
├── Credit_Card_Approval/
│   ├── Application_Data.csv
│   └── Credit Card Application Approval .ipynb
├── Motor_Claims/
│   ├── French_Motor_Project.ipynb
│   └── README.md
└── README.md
```

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for notebooks formatting rules.

## License
Distributed under the MIT License. See `LICENSE` for details.

## Contact
Eswar Melam - [LinkedIn](https://linkedin.com/in/eswar-melam) - eswar.melam@gmail.com
