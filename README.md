# Project-Samantha



A full-stack enterprise care analytics platform built entirely in Python with Streamlit. Project Samantha provides a comprehensive solution for care facilities to move from anecdotal guesswork to data-driven, evidence-based decision-making.



---
## The Problem

Care facilities often struggle with scattered data across spreadsheets and paper files, making it difficult to make objective decisions. This leads to hidden inefficiencies, wasted resources, and challenges in proving the effectiveness of their programs to stakeholders like funders and regulators.

---
## The Solution

Project Samantha solves this by providing a single, centralized platform to manage the entire data lifecycle. It empowers managers, clinicians, and administrators with the tools to track costs, measure client outcomes, and optimize their services for the best results.

---
## Key Features

* **Secure Role-Based Authentication**: Different views and permissions for Administrators, Supervisors, and Staff. Includes a full password reset workflow.
* **Interactive Dashboards**: Dynamic, filterable dashboards for high-level executive overviews, detailed individual progress tracking, and intervention analysis.
* **End-to-End Data Management**:
    * **Smart Data Entry**: Validated forms to ensure data integrity at the point of entry.
    * **Data Quality Toolkit**: An interactive tool to find and correct real-world data issues like duplicates, invalid entries, and future-dated records.
    * **Bulk Operations**: UI for importing and updating records from CSV files.
* **Predictive Analytics**: A machine learning pipeline using **XGBoost** to classify and predict future client outcomes ('Low', 'Medium', or 'High'), enabling proactive care. The model is tuned and trained on-demand directly within the app.
* **Comprehensive Reporting**: A multi-faceted reporting engine with the ability to generate and export professional **PDF** and **CSV** reports for stakeholders.
* **Full Admin Backend**: A complete administrative panel to manage:
    * Users and permissions.
    * Financial settings, including budgeting and cost allocation.
    * Clinical configurations, such as custom outcome metrics, assessment schedules, and standardized protocols.

---
## Built With

This project is a testament to the power of a modern data science stack:

* **Framework**: [Streamlit](https://streamlit.io/)
* **Database**: [DuckDB](https://duckdb.org/) (in-memory & file-based)
* **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
* **Data Visualization**: [Plotly](https://plotly.com/)
* **Machine Learning**: [Scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.ai/)
* **PDF Generation**: [FPDF2](https://github.com/py-pdf/fpdf2)

---
## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

You need to have Python 3.8+ installed on your system.

### Installation

1.  **Create a `requirements.txt` file**: Before uploading to GitHub, run this command in your terminal to capture all the libraries your project needs:
    ```sh
    pip freeze > requirements.txt
    ```
2.  **Clone the repo**:
    ```sh
    git clone https://github.com/AlexBiobelemo/Project-Samantha
    ```
3.  **Navigate to the project directory**:
    ```sh
    cd project-samantha
    ```
4.  **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```
5.  **Run the Streamlit app**:
    ```sh
    streamlit run app.py
    ```

---



