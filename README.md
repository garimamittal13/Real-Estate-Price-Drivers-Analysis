# Real Estate Price Drivers Analysis for Indian Property Listings

## Overview
This project analyzes Indian real-estate listing data to understand what drives property prices, identify missing-data issues, flag suspicious records, and build a baseline price prediction model.

## Problem Statement
Property listing datasets are often noisy and incomplete. This project focuses on:
- cleaning messy listing data
- analyzing missing values and suspicious records
- understanding key factors associated with price
- building a baseline model for price estimation

## Dataset
Kaggle real-estate dataset containing property prices, area/size, location fields, and listing attributes.

## Method
1. Cleaned column names and removed duplicates
2. Converted price and area fields to numeric
3. Generated missing-value and anomaly reports
4. Performed exploratory data analysis
5. Trained a Random Forest regression baseline
6. Measured MAE, RMSE, and R²
7. Computed feature importance

## Key Outputs
- missing value report
- suspicious listings report
- correlation heatmap
- price distribution plots
- median price by city
- feature importance chart
- baseline regression metrics

## Business Relevance
This mirrors real-world property data workflows:
- raw data cleaning
- data quality diagnostics
- market segmentation
- price driver analysis
- baseline valuation modeling

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## How to Run
```bash
pip install -r requirements.txt
python src/run_analysis.py
