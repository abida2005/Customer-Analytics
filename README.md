# ShoplyticsHub – Customer Analytics System

## 📌 Project Overview

ShoplyticsHub is a Django-based Customer Analytics System designed to help shopkeepers understand customer behavior using transaction data.

The system provides customer analytics through RFM analysis, customer segmentation, churn prediction, cohort analysis, interactive visualizations, and downloadable PDF reports.

## 🎯 Problem Statement

Small shopkeepers often have transaction data but do not have an easy way to analyze customer behavior.

ShoplyticsHub provides a simple web-based platform where shopkeepers can upload transaction datasets and generate meaningful customer insights without requiring advanced technical knowledge.

## ✨ Features

- 📂 Dataset Upload
- 👥 Customer Management
- 📊 RFM Analysis
- 🎯 Customer Segmentation
- 🔄 Churn Prediction
- 📈 Cohort Analysis
- 📊 Interactive Dashboard
- 📄 PDF Report Generation
- 🏪 Multi-Shop Support
- 🔍 Transaction Management
- 🚫 Duplicate Dataset Detection

## 🛠️ Technologies Used

### Backend
- Python
- Django

### Data Analysis & Machine Learning
- Pandas
- Scikit-learn
- NumPy

### Frontend
- HTML
- CSS
- Bootstrap
- Chart.js

### Database
- SQLite

### Reporting
- ReportLab

## ⚙️ How the System Works

1. The shopkeeper logs into the system.
2. Transaction data is uploaded as a CSV file.
3. The system validates and stores the dataset.
4. Customer transaction data is processed using Python and Pandas.
5. RFM analysis is performed to understand customer behavior.
6. Customers are segmented based on their purchasing patterns.
7. Churn prediction identifies customers who may stop purchasing.
8. Cohort analysis helps understand customer retention.
9. Results are displayed through interactive dashboards.
10. Analytical results can be generated as PDF reports.

## 📊 Analytics

### RFM Analysis

The system analyzes customers using:

- Recency
- Frequency
- Monetary Value

Customers are grouped into meaningful segments such as:

- Champions
- Loyal Customers
- At Risk
- Lost Customers

### Churn Prediction

A machine-learning model is used to identify customers who may be at risk of churning.

### Cohort Analysis

Cohort analysis is used to study customer retention and purchasing behavior over time.

## 🖥️ Screenshots

Screenshots of the application will be added here.

## 📁 Project Structure

```text
Customer-Analytics/
│
├── accounts/
├── app_name/
│   ├── static/
│   └── templates/
│
├── customer_analytics/
├── reports/
│
├── manage.py
├── requirements.txt
├── .gitignore
└── README.md