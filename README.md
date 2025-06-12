# Breast Cancer Risk Prediction project

## Executive Summary  
This project develops a Streamlit-based web application to predict risk and treatment outcomes for breast cancer patients using machine learning on demographic, clinical, and genomic data. We will train XGBoost and Random Forest classifiers on age, tumor characteristics, and gene-expression features to:  
1. **Predict individual risk categories** (low/medium/high) based on demographic and tumor data.  
2. **Predict treatment-specific outcomes** (e.g., response vs. non-response) from gene-expression profiles.  
3. **Provide interactive data exploration** across two public datasets.  
4. **Offer lifestyle-and-diet guidance** via curated articles and videos.  

By integrating predictive analytics with an easy-to-use interface, the app will support clinicians and patients in personalized decision-making.

---

## Motivation  
Breast cancer’s heterogeneity makes outcome prediction and risk stratification difficult. Most existing tools either focus narrowly on clinical features or ignore high-dimensional genomic data. A combined approach—using both demographic/tumor characteristics and gene-expression—can yield more accurate, actionable insights. Building this tool in Streamlit ensures rapid prototyping and easy deployment, while curated lifestyle resources empower patients beyond clinical metrics.

---

## Data Questions  
1. **Risk Prediction**  
   - How accurately can age, demographic factors, and tumor characteristics classify patients into low/medium/high risk groups?  
2. **Treatment Outcome Prediction**  
   - Which gene-expression signatures best distinguish responders from non-responders to specific therapies?  
3. **Model Comparison**  
   - How do XGBoost and Random Forest classifiers compare in performance on both tasks?  
4. **Data Exploration**  
   - What key patterns and correlations emerge in clinical vs. genomic variables across datasets?  
5. **Patient Guidance**  
   - Which lifestyle and dietary recommendations correlate with improved outcomes, and how can they be effectively presented?

---


## Minimum Viable Product  
1. **Trained Models**  
   - XGBoost & Random Forest classifiers for:  
     - **Risk category** (demographics & tumor features)  
     - **Treatment response** (gene-expression inputs)  
2. **Streamlit Web App** with four tabs:  
   1. **Risk Factor Prediction**  
      - Form inputs: age, demographic, tumor size/grade/ER/PR/HER2 status  
      - Outputs: risk category and feature-importance explanation  
   2. **Treatment Outcome Prediction**  
      - Gene-expression uploader or selection  
      - Outputs: predicted responder vs. non-responder probabilities  
   3. **Data Analysis**  
      - Interactive charts, tables, and statistical summaries from two datasets  
   4. **Lifestyle & Diet Guidance**  
      - Curated articles & YouTube links on exercise, nutrition, survivorship  

---

## Data Sources  
- **METABRIC** https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric. Data contain 1,992 gene expression from primary breast tumours and clinical follow up information.
Gene expression

Platform: Illumina HT-12 v3 microarray

~24,368 probes (genes) measured in 1,904–1,992 samples (after QC/filtering) 
researchgate.net

Copy-number profiles

Platform: Affymetrix SNP 6.0 arrays

Segmented (CBS & HMM) copy-number aberrations for both discovery and validation cohorts 
ega-archive.org

Somatic mutations

Panel of ~175 breast-cancer-relevant genes

Available for ~2,433 patients in cBioPortal summary 
cbioportal.org

Clinical annotations

Demographics: age at diagnosis, ethnicity

Tumor features: size, grade, lymph node status, ER/PR/HER2 status

Treatments: surgery type, adjuvant therapy

Outcomes: overall survival (OS), breast-cancer-specific survival (BCSS) 

  - Gene expression, copy-number variants, clinical & survival data  
- Breast Cancer Surveillance Consortium https://www.bcsc-research.org/index.php/datasets/rf
The data were collected during January 2005 – December 2017 from 6,788,436 mammograms at different community breast imaging practices
Key variables collected:
Demographic: 
Age (in 5-year groups)
1 = Age 18-29
2 = Age 30-34
3 = Age 35-39
4 = Age 40-44
5 = Age 45-49
6 = Age 50-54
7 = Age 55-59
8 = Age 60-64
9 = Age 65-69
10 = Age 70-74
11 = Age 75-79
12 = Age 80-84
13 = Age >85
Race/ethnicity	
1 = Non-Hispanic white
2 = Non-Hispanic black
3 = Asian/Pacific Islander
4 = Native American
5 = Hispanic
6 = Other/mixed


Age when had the first period
0 = Age >14
1 = Age 12-13
2 = Age <12


Age (years) at first birth	
0 = Age < 20
1 = Age 20-24
2 = Age 25-29
3 = Age >30
4 = Nulliparous



Family/personal history: First-degree family history of breast cancer; personal history of breast biopsy or cancer
0 = No
1 = Yes


Breast characteristics: BI-RADS breast density
1 = Almost entirely fat
2 = Scattered fibroglandular densities
3 = Heterogeneously dense
4 = Extremely dense


Hormone use
0 = No
1 = Yes


menopausal status: 
1 = Pre- or peri-menopausal
2 = Post-menopausal
3 = Surgical menopause

Body size: BMI group
1 = 10-24.99
2 = 25-29.99
3 = 30-34.99
4 = 35 or more


Biopsy history
0 = No
1 = Yes


Breast cancer history
0 = No
1 = Yes

 
