# Bank Customer Churn Analysis — SQL + Python (ML) + Power BI

An end-to-end churn analysis project on a bank customer dataset. Data is loaded and explored in **MySQL**, churn is predicted using **Python (scikit-learn + SHAP)**, and results are visualized in an interactive **Power BI** dashboard.

## Dashboard preview

![Churn Dashboard]

<img width="1026" height="711" alt="Screenshot 2026-08-06 222043" src="https://github.com/user-attachments/assets/d6602cb4-2833-4308-bceb-443281c96f8a" />

## Project overview

Customer churn is one of the highest-leverage problems in banking — losing a customer costs far more than retaining one. This project answers:

- What's the overall churn rate, and how does it vary by country, gender, and age group?
- Which customer segments are highest risk (high balance, low engagement)?
- Can churn be predicted before it happens, and which features drive that prediction?
- How much account balance is at risk from customers likely to churn?

## Tech stack

| Layer | Tool |
|---|---|
| Database | MySQL 8.0 |
| Data loading | `LOAD DATA INFILE` |
| SQL analysis | Views, window functions, CASE-based segmentation |
| Machine learning | Python — pandas, scikit-learn, SHAP |
| Visualization | Power BI Desktop |

## Dataset

`Bank Customer Churn Prediction.csv` — 10,000 customers, no missing values.

```
customer_id, credit_score, country, gender, age, tenure, balance,
products_number, credit_card, active_member, estimated_salary, churn
```

Loaded directly into a single `customers` table via `LOAD DATA INFILE` (schema in [`Churn_Analysis.sql`](Churn_Analysis.sql)) — this dataset didn't need the multi-table normalization the Superstore project did, since it's already one row per customer with no repeating groups.

## SQL analysis

All queries are in [`Churn_Analysis.sql`](Churn_Analysis.sql). Highlights:

- **Churn rate by country and gender** — grouped breakdown, sorted by highest churn rate, to spot which segments need attention first
- **Top churn segment by country** — `RANK() OVER (ORDER BY SUM(churn) DESC)` to rank countries by total churned customers
- **`CHURN_BY_AGE` view** — buckets customers into Young/Middle/Senior with `CASE`, and computes each group's churn rate *and* its contribution to total churn using a window function (`SUM(churn) OVER ()`) — this distinguishes "this group churns a lot" from "this group is responsible for most of our churn," which are different questions with different fixes
- **Customer risk segmentation** — a `CASE`-based rule splitting customers into Churned / High Risk / Medium Risk / Low Risk based on balance and activity status, as a rule-based baseline before the ML model
- **High-value churned customers** — filters to customers who already churned with balance and salary both above 100,000, the highest-value losses to review first
- **`high_risk_customers` view** — flags currently-active customers with high balance but low credit score, a proactive early-warning segment
- **`kpi` view** — total customers, churned count, churn rate, average balance, average credit score — a single clean summary source

## Machine learning (Python)

Full pipeline in `churn_analysis.py`:

1. **EDA** — churn rate by country, gender, active-member status, and number of products
2. **Feature engineering** — age buckets, has-balance flag, one-hot encoded country/gender
3. **Baseline model**: Logistic Regression (`class_weight='balanced'` to handle the ~20% churn imbalance)
4. **Random Forest** — outperformed the baseline, ROC-AUC 0.86
5. **SHAP explainability** — identifies `age`, `products_number`, and `active_member` as the strongest churn drivers
6. **Business impact translation** — flags high-risk customers (churn probability ≥ 0.5) and quantifies total account balance at risk
7. Outputs: `churn_predictions.csv` (every customer scored, with `customer_id`, features, probability, and prediction) and `top_50_at_risk_customers.csv` (prioritized retention list)

### Model results

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.78 |
| Random Forest | **0.86** |

## Power BI dashboard

"Bank Customer Churn Analysis" includes:

- **KPI cards**: Total Customers, Churn Rate, Predicted Churn, Balance at Risk
- **Country and gender slicers**
- **Feature importance** chart (from the SHAP/model output)
- **Customer volume by risk level** donut (Low / Medium / High)
- **Churn rate by number of products** — the strongest single predictor found in EDA
- **Churn rate by country**
- **Retention trends by age group**
- **Top at-risk customers table** — the actionable retention list, sorted by churn probability

Data source: `churn_predictions.csv` (Python model output), loaded directly — no relationships needed since it's a single flat table.

## How to reproduce

1. Load the schema and data:
   ```sql
   mysql -u root -p < Churn_Analysis.sql
   ```
   (Update the `LOAD DATA INFILE` path to your local CSV location.)
2. Run the ML pipeline:
   ```bash
   pip install pandas scikit-learn shap matplotlib seaborn
   python churn_analysis.py
   ```
3. In Power BI Desktop: **Get Data → Text/CSV** → load `churn_predictions.csv` (and `top_50_at_risk_customers.csv` for the action-list table)
4. Rebuild the visuals following the dashboard section above, or open the included `.pbix` if shared.

## Repository contents

```
Churn_Analysis.sql              -- schema, data load, and SQL analysis queries
churn_analysis.py               -- EDA, feature engineering, modeling, SHAP, business impact
README.md                       -- this file
dashboard_screenshot.png        -- dashboard preview image (add your own)
churn_predictions.csv           -- model output (all customers, probability + prediction)
top_50_at_risk_customers.csv    -- prioritized retention list
Bank Customer Churn Prediction.csv -- raw dataset
```

## Key insights

- **Customers with 3-4 products churn dramatically more** (83% and 100% respectively) than customers with 1-2 products (~14-8%) — a strong, non-linear signal worth investigating further given the small sample size at 3-4 products.
- **Inactive members churn at nearly twice the rate of active members** — the most actionable lever, since "increase engagement" is a concrete retention strategy.
- **Germany has a notably higher churn rate** than France or Spain, though this weakens once other features are controlled for in the model — likely correlated with age/product mix rather than an independent geography effect.
- **Older, higher-balance customers churn more** — counterintuitive, since higher-balance customers are often assumed to be "stickier."

## Author

Faiez — final-year engineering student, IIT Indore.
