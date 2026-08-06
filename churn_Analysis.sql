CREATE DATABASE churn;
USE churn;

CREATE TABLE customers (
    customer_id BIGINT PRIMARY KEY,
    credit_score INT,
    country VARCHAR(50),
    gender VARCHAR(10),
    age INT,
    tenure INT,
    balance DECIMAL(15,2),
    products_number INT,
    credit_card TINYINT,
    active_member TINYINT,
    estimated_salary DECIMAL(15,2),
    churn TINYINT
);

SHOW VARIABLES LIKE 'secure_file_priv';

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Bank Customer Churn Prediction.csv'
INTO TABLE customers
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- churn rate by multiple factors
SELECT 
   country,
   gender,
   COUNT(*) AS Total_Customers,
   SUM(churn) AS churned,
   ROUND(SUM(churn)*100.0/COUNT(*),2) AS churn_rate
FROM customers
GROUP BY country , gender
ORDER BY churn_rate DESC;

-- Top churn Segment
SELECT 
    country,
    COUNT(*) AS total,
    SUM(churn) AS churned,
    RANK() OVER (ORDER BY SUM(churn) DESC) AS churn_rank
FROM customers
GROUP BY country;

-- Churn by age group
CREATE VIEW CHURN_BY_AGE AS
SELECT 
    CASE 
        WHEN age < 30 THEN 'Young'
        WHEN age BETWEEN 30 AND 50 THEN 'Middle'
        ELSE 'Senior'
    END AS age_group,
    
    COUNT(*) AS total,
    SUM(churn) AS churned,
    
    ROUND(SUM(churn)*100.0/COUNT(*),2) AS churn_rate,
    
    ROUND(
        SUM(churn)*100.0 / SUM(SUM(churn)) OVER (),
        2
    ) AS contribution_pct

FROM customers
GROUP BY age_group;

-- CUSTOMER RISK SEGMENTATION
SELECT *,
    CASE 
        WHEN churn = 1 THEN 'Churned'
        WHEN balance > 100000 AND active_member = 0 THEN 'High Risk'
        WHEN balance > 50000 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END AS risk_category
FROM customers;


-- HIGH VALUE CUSTOMERS WHO CHURNED
SELECT *
FROM customers
WHERE churn = 1
AND balance > 100000
AND estimated_salary > 100000
ORDER BY balance DESC;

-- High Risk Customers
CREATE VIEW high_risk_customers AS
SELECT *
FROM customers
WHERE churn = 0
AND balance > 100000 
AND credit_score < 500;


-- KPI
CREATE VIEW kpi AS
SELECT 
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(SUM(churn)/COUNT(*) * 100, 2) AS churn_rate,
    AVG(balance) AS avg_balance,
    AVG(credit_score) AS avg_credit_score
FROM customers;

