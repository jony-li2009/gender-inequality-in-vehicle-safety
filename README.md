# Gender Inequality and Fatality Risk in Traffic Crashes

## Overview
**Plain language:**  
This project studies whether men and women face different risks of dying in traffic crashes, even when crashes happen under similar conditions. Using large-scale U.S. crash data, the analysis shows that gender-based differences in fatality risk persist and vary across vehicle types and crash scenarios.

**Technical:**  
We analyze traffic crash records using statistical risk models with interaction terms to quantify gender-specific effects on fatality outcomes while controlling for vehicle, environmental, and crash-related factors.

---

## Motivation
**Plain language:**  
Most traffic safety research looks at overall risk factors like speed or road conditions but does not carefully examine whether safety outcomes differ by gender. Because vehicles and safety systems are often designed using average body models, this can unintentionally disadvantage certain populations.

**Technical:**  
This work investigates whether gender acts as an independent risk factor or interacts with other variables (e.g., vehicle type, crash configuration) to influence fatality probability.

---

## Data
**Plain language:**  
The study uses real traffic crash records collected across the United States, including information about drivers, vehicles, and crash outcomes.

**Technical:**  
The dataset includes:
- Driver demographics (including gender)
- Vehicle characteristics
- Crash environment and configuration
- Injury severity and fatality indicators

Records with missing or inconsistent critical variables are excluded during preprocessing.

---

## Methodology
**Plain language:**  
We clean the data, compare crash outcomes between groups, and use mathematical models to estimate how likely fatal outcomes are under different conditions.

**Technical:**  
1. **Preprocessing**
   - Data cleaning and filtering
   - Encoding categorical variables
   - Feature normalization where appropriate

2. **Modeling**
   - Logistic regression–based fatality risk models
   - Gender interaction terms to capture differential effects
   - Stratified analyses by vehicle type and crash context

3. **Evaluation**
   - Comparison of estimated fatality probabilities
   - Statistical significance testing of interaction effects

---

## Key Findings
**Plain language:**  
- Women face higher fatality risk than men in certain crash scenarios.
- The difference is not constant and depends on vehicle type and crash conditions.
- Some safety systems appear less effective for specific populations.

**Technical:**  
- Gender coefficients and interaction terms are statistically significant in multiple model specifications.
- Risk disparities vary by vehicle category and crash environment.
- Results suggest systematic bias in safety performance rather than random variation.

---

## Figures
All figures from the paper are included in this repository.

Example:
```md
![Fatality Risk by Gender](figures/fatality_risk_gender.png)
