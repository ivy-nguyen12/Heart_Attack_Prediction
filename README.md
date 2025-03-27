# Heart Attack Risk Prediction

## Overview
This project aims to predict heart attack risk using health, lifestyle, and demographic factors through various machine learning models. By analyzing a simulated dataset across 24 variables and multiple countries, we built models to understand patterns and potential prevention strategies for cardiovascular disease.

Prepared by: Vy Nguyen, Yanan Sun, Becky Wang, Xinying Wu

Key results include:
- SVM (Tuned) achieved the highest global accuracy of **64%**
- KNN performed **well on country subsets**, especially Italy and Japan
- Random Forest, despite tuning, showed strong overfitting
- Key features varied by country, highlighting the need for regional strategies

## Dataset
- **Source:** AI-generated simulated dataset
- **Size:** 8,763 records; 24 features in original data, 52 after dummy encoding
- **Scope:** Includes patients across multiple countries and continents
- **Two Versions:**
  - `data`: Categorical features as factors
  - `data1`: Dummy-encoded for numerical modeling

### 📋 Data Dictionary

This dataset contains **24 health and demographic features** organized into four key categories. The definitions and descriptions for each variable are provided in the `Final_Data_Dictionary.pdf` file located in the `Reports/` folder.

- **6 Demographic Factors**:  
  `Sex`, `Age`, `Country`, `Continent`, `Hemisphere`, `Income`
- **6 Physiological & Clinical Variables**:  
  `Cholesterol`, `Blood.Pressure (Systolic, Diastolic)`, `Triglycerides`, `Heart.Rate`, `BMI`, `Obesity`
- **5 Medical History Factors**:  
  `Diabetes`, `Family.History`, `Previous.Heart.Problems`, `Medication.Use`, `Stress.Level`
- **7 Lifestyle Factors**:  
  `Smoking`, `Alcohol.Consumption`, `Diet`,  
  `Exercise.Hours.Per.Week`, `Physical.Activity.Days.Per.Week`,  
  `Sedentary.Hours.Per.Day`, `Sleep.Hours.Per.Day`

## Tools & Methodology Overview
**Languages & Libraries:** R (ggplot2, rpart, e1071, randomForest, caret)

### Data Preprocessing:
- Encoded binary and multi-category variables
- Created dummies for categorical features for specific models
- Engineered `Systolic` and `Diastolic` from `Blood.Pressure`
- Exported both factor-based and dummy-encoded datasets

### Exploratory Analysis:
- Distribution of risk across gender, age, income, country
- Correlation matrices by factor groups (demographic, lifestyle, etc.)

### Modeling Techniques:
- Logistic Regression (global & by country)
- Naive Bayes (global & by country)
- KNN (Min-Max & Z-Score normalization, regional tuning)
- Decision Tree (pruning & country-specific)
- Random Forest (feature tuning, importance analysis)
- SVM (global and regional, parameter tuning)

### Evaluation Metrics:
- Accuracy on train/test splits
- Confusion matrices for classification performance
- Country-specific breakdowns

## Highlighted Visualizations

**Heart Attack Risk by Country:**
![Country Risk](Notebooks/heart_attack_by_country.png)

**Overall Heart Attack Risk Distribution:**
![Risk Pie Chart](Notebooks/risk_distribution_pie.png)

**Feature Importance from Random Forest:**
![Feature Importance](Notebooks/feature_importance_rf.png)

## Results & Key Insights
- **SVM outperformed** other models on the global dataset (~64% test accuracy)
- **KNN was most effective** for Italy and Japan; accuracy up to ~66%
- Logistic Regression and Naive Bayes showed **limitations due to assumptions**
- Random Forest models consistently overfit, with 100% training accuracy
- Key regional predictors:
  - Italy: Obesity, BMI, Medication Use
  - Japan: Heart Rate, Diet, Medication Use
  - US: Income, Physical Activity, Triglycerides

## Key Deliverables
- R script: `Heart_Attack_Risk_Prediction.R`
- Final presentation (PDF)
- Appendix and data dictionary (PDF)
  
## What I Learned
- Model assumptions matter: Naive Bayes and Logistic Regression underperformed due to oversimplified assumptions
- SVMs can capture non-linear patterns better, but tuning is crucial
- Data quality and balance are essential; simulated data can lead to model bias
- Dummy encoding is necessary for algorithms like KNN and SVM

## What I Plan to Improve
- Test on real-world healthcare datasets (e.g., UCI Heart Disease)
- Explore deep learning or ensemble stacking models
- Integrate time-based features for longitudinal health analysis
- Conduct deeper regional analysis across more countries

## About Me
Hi, I’m Vy Nguyen and I’m currently pursuing my MS in Business Analytics at UC Irvine. I’m passionate about data analytics in Finance and Healthcare. Connect with me on [LinkedIn](https://www.linkedin.com/in/vy-ngoc-lan-nguyen).
