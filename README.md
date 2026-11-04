# Delhivery Last-Mile Delay Intelligence Pipeline

This repository contains an end-to-end Machine Learning pipeline designed to address last-mile delivery delays across Delhivery's regional hub network[cite: 1]. The system cleans raw shipment logs, forecasts delivery durations, predicts potential service promise breaches, and clusters hub operations into actionable risk playbooks[cite: 1, 8, 11, 13, 16].

## 1. How to Run

To ensure complete reproducibility across clean environments, execute the scripts sequentially using Python 3.10+[cite: 2].

### Environment Setup

# Clone repository and navigate to root directory
git clone [https://github.com/your-username/delhivery-delay-intelligence.git](https://github.com/your-username/delhivery-delay-intelligence.git)
cd delhivery-delay-intelligence

# Install exact pinned dependencies
pip install -r requirements.txt


### Execution Pipeline Sequence

Run the pipeline in the following exact order:

# Step 0: Generate initial synthetic datasets (creates hubs.csv, raw_shipments.csv, new_shipments.csv)
python generate_data.py

# Step 1: Part A - Data cleaning, feature engineering, and EDA checks
python part_a_clean_eda.py

# Step 2: Part B - Train and evaluate delivery-hours regression models
python part_b_regression.py

# Step 3: Part C - Train delay classifiers and select operational alert threshold
python part_c_classification.py

# Step 4: Part D - Perform hub segmentation and save model artifacts
python part_d_segments.py

# Step 5: Package model for production deployment
python package_model.py

# Step 6: Score incoming dispatches and generate operations action list
python predict_delay.py



Note: Running `package_model.py` regenerates `models/delay_model.pkl` locally. The committed pickle in `models/` relies strictly on `scikit-learn==1.5.2` as pinned in `requirements.txt`.



## 2. Data Findings

Analysis of the historical shipment logs (`data/modelling_table.csv`) yielded key observations regarding last-mile performance:

1. **Service Type Performance:** Express shipments exhibit the highest delay rate at **48.97%** (119 delayed out of 243 shipments), compared to Standard at **20.56%** (132 / 642) and Economy at **8.57%** (27 / 315). Tight operational windows in express fulfillment make it highly sensitive to route disruptions.


2. **Hub Tier Delay Scaling:** Delay rates scale sharply across official tier labels: Tier 1 hubs average a **4.00%** delay rate, Tier 2 hubs average **21.50%**, and Tier 3 hubs reach **44.00%**.


3. **Distance Band Dynamics:** Delay risk increases non-linearly with distance band:


* `0-200 km`: 2.19% delay rate


* `201-400 km`: 13.68% delay rate


* `401-600 km`: 43.64% delay rate


* `601+ km`: 85.23% delay rate




4. **Top Correlated Features with Delivery Hours:** Excluding `delivery_hours` itself, the top three continuous features positively correlated with delivery hours are `promised_hours` (0.8370), `distance_km` (0.4010), and `n_stops` (0.3845).


5. **Log-Transformation Justification:** `declared_value_inr` exhibits severe right skewness with a raw mean of **INR 1,533.05** versus a median of **INR 1,121.50**. Applying a natural log transformation (`log_declared_value`) yields a near-symmetrical distribution with mean **7.028** and median **7.022**, stabilizing variance for linear modeling.



## 3. Model Results (Regression)

### Part B — Regression (`delivery_hours`)

* **Baseline Model:** Predicting average training duration yields a validation $R^2$ of **-0.021** with an RMSE of **19.82 hours**, reflecting zero predictive utility.


* **Model Comparison:**
* *Linear Regression:* Validation $R^2 = 0.931$, RMSE = 5.21 hrs


* *Gradient Boosting Regressor:* Validation $R^2 = 0.931$, RMSE = 5.18 hrs


* *Random Forest Regressor:* Validation $R^2 = 0.914$, RMSE = 5.79 hrs


* *Decision Tree Regressor (Depth 6):* Validation $R^2 = 0.887$, RMSE = 6.64 hrs




* **High-Bias vs. High-Variance Analysis:**
* *Depth 1:* High-bias regime ($\text{val\_rmse} - \text{train\_rmse} = 0.50$, but overall error remains high near 12.30 hrs).


* *Unlimited Depth:* High-variance regime with zero training error ($\text{train\_rmse} = 0.00$) but severe validation degradation ($\text{val\_rmse} = 7.79$, gap = 7.79 hrs).


* *Optimal Depth:* Depth $d=6$ minimizes validation error ($\text{val\_rmse} = 6.28$).




* **Learning Rate Tuning:**
* `lr = 0.01`: Underfits due to slow step progression ($R^2 = 0.713$).


* `lr = 0.10`: Optimal step convergence ($R^2 = 0.931$).


* `lr = 1.00`: Overshoots minimum with unstable updates ($R^2 = 0.844$).




* **Cross-Validation & Final Evaluation:** Both Linear Regression and Gradient Boosting achieved mean 5-fold cross-validation $R^2$ scores of **0.926** and **0.935** respectively. **Gradient Boosting** was selected for production, achieving a final Test Set $R^2$ of **0.949**.


## 4. Model Results (Classification)

### Part C — Classification (`is_delayed`)

* **Baseline & Un-tuned Models:**
* *Blind Baseline:* Classifying all parcels as "on time" yields **76.67%** accuracy but an $F_1$-score of **0.00**.


* *Logistic Regression:* Validation $F_1 = 0.805$.


* *Random Forest (17 features):* Validation $F_1 = 0.800$.




* **Feature Selection & Noise Filtering:** The 5 synthetic noise columns (`label_print_batch`, `scanner_build_no`, `tape_roll_no`, `sticker_serial_mod`, `shelf_slot_no`) contributed **22.5%** of total feature importance in the unconstrained forest. Using `SelectKBest(f_classif, k=7)` filters noise features completely and improves validation performance.


* **Threshold Selection:** Evaluating precision/recall tradeoffs on validation data under the operational rule (optimizing for recall $\ge 0.80$) led to selecting threshold $t = 0.30$.


* **Test Set Evaluation:** Evaluating on 180 unseen test rows (containing 41 actual delayed parcels) at threshold $t = 0.30$:


* **Precision:** 0.689


* **Recall:** 0.756


* **$F_1$-Score:** 0.721





## 5. Operations Recommendation

### Tier vs. Segment Mismatches

Applying K-Means clustering ($K=3$) on hub metrics (`mean_delivery_hours`, `delay_rate`, `mean_distance_km`, `mean_n_stops`) reveals that 4 out of 24 hubs operate out of band relative to their designated tier:

| Hub ID | City | Official Tier | Cluster Segment | Operational Impact |
| --- | --- | --- | --- | --- |
| **HUB07** | Kolkata | Tier 1 | **Watch** | Performs like a regional hub; requires monitoring.

 |
| **HUB13** | Nagpur | Tier 2 | **Escalate** | Severe delay profile matching stretched hubs.

 |
| **HUB15** | Kochi | Tier 2 | **Escalate** | High stop density causing bottleneck delays.

 |
| **HUB22** | Dehradun | Tier 3 | **Watch** | Outperforms tier expectations; needs reduced buffer targets.

 |

### Operational Alert Rule & Value

* **Hub Playbooks:**
* **Escalate:** Hubs with high baseline delay rates. Flagged parcels trigger direct hub manager escalations.


* **Watch / Standard:** Flagged parcels trigger proactive customer notification calls.


* **No Action:** Unflagged parcels proceed through standard dispatch.




* **Scoring Results on New Dispatches:** Out of 48 incoming parcels in `data/new_shipments.csv`, **14 parcels** were flagged for delay risk. Among flagged parcels, **6** were routed for hub manager escalations (`Escalate` hubs), while **8** triggered proactive customer notification calls.


* **K-Means Stability Justification:** $K=3$ was maintained because $K=2$ fails to isolate high-risk hubs, whereas $K=4$ or $K=5$ produces micro-clusters with fewer than 6 hubs, reducing operational scalability.




## 6. Production Plan

### Mapped Production Characteristics

1. **Accuracy & Reliability:** Modular pipeline design using scikit-learn standard components with full reproducibility checks.


2. **Reproducibility:** Seed pinning (`random_state=42`) across data splits, feature selection, and model training ensures identical execution.


3. **Speed & Efficiency:** Inference completes in under 200 milliseconds for batches of 500 dispatches.
4. **Maintainability:** Shared preprocessing functions in `src/shipment_prep.py` eliminate code drift between training and scoring.



### Monitoring Metrics & Triggers

* **Weekly Delay Rate:** Alert triggered if overall delay rate exceeds historical baseline by > 5%.


* **Flagged Volume Ratio:** Alert triggered if daily flagged share exceeds 35% of total dispatches.


* **Recall Drift:** Evaluated monthly on completed deliveries; retrain triggered if recall drops below 0.70.



### Retraining Schedule & Sale Window Strategy

* **Retraining Schedule:** Bi-weekly batch retraining on rolling 90-day delivery logs.
* **Handling Sale Events (`sale_event_flag`):** The September 15–28 festival sale window causes localized volume spikes. Rather than creating a separate model, `sale_event_flag` is encoded directly as a feature, allowing the model to dynamically lift baseline delay probabilities during promotional events without over-predicting delays during normal weeks.




## 7. Limitations

1. **Synthetic Data Boundaries:** The dataset is generated synthetically; real-world logistics introduce unmodeled external factors such as weather extremes and traffic disruptions.


2. **Small Validation/Test Sample Size:** With 180 validation rows and 41 delayed test cases, performance estimates carry a variance margin.


3. **Threshold Tuning Constraints:** Operational thresholds tuned on a single validation split may require dynamic seasonal recalibration in live production deployment.


