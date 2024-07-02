# Salary Prediction - Python

This project is a Salary Predictor application developed using Python. It analyzes IT salary data and predicts salaries based on user inputs through a graphical user interface (GUI).

## Requirements

- Python 3.x
- Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, tkinter

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sushan4/salary_predict.git
    cd salary_predictor
    ```

2. Install the required libraries:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```

## Usage

1. Run the main script:
    ```bash
    python salary_predictor.py
    ```

2. Use the GUI to input details:
    - Enter years of experience.
    - Select job level (Head, Middle, Senior).
    - Select company size (50-100, 100-1000, More Than 1000).
    - Select company type (Agency, Product, Startup).

3. Click "PREDICT" to get the predicted salary.

## Project Structure

- `it_salary.csv`: Dataset containing IT salary information.
- `salary_predictor.py`: Main script with data analysis, model training, and GUI code.

## Summary

This project provides a simple application to predict IT salaries based on various factors using a linear regression model and a user-friendly GUI.
