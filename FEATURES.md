# Project Samantha - A Deep Dive into All Features

This document provides an exhaustive, granular overview of every feature and sub-system within **Project Samantha**. The application is architected as a full-stack enterprise solution, providing an end-to-end, interactive platform for care analytics that rivals professional-grade software.

---
## I. Core Architecture & Design Philosophy

This section details the foundational principles that ensure the application is robust, performant, and maintainable, meeting high standards for code quality and design.

* **Modular & Maintainable Codebase:** The application is organized into logical, single-responsibility functions (e.g., `show_executive_dashboard`, `perform_predictive_modeling`), making the code clean, readable, and easy to update.
* **Centralized Data Source:** A single **DuckDB** database acts as the single source of truth, eliminating data silos and ensuring consistency across all modules.
* **Performance-First UX:**
    * **On-Demand Model Training:** The computationally expensive machine learning model is trained only when explicitly triggered by an administrator, preventing UI lag during normal use.
* **Robust Error Handling:** Every major user-facing function is wrapped in `try...except` blocks to gracefully handle unexpected issues and provide clear feedback without crashing the application.

---
## II. Security & Access Control

A multi-layered security model to protect sensitive data and ensure users only access information relevant to their roles.

* **Role-Based Access Control (RBAC):**
    * **Administrator:** Unrestricted access. Can view all data, manage users, configure system settings, and train the predictive model.
    * **Supervisor:** Can view team-level data and reports but is restricted from system-wide configuration and other administrative tasks.
    * **Staff:** Highly restricted view. Can only access their personal dashboard and log data for individuals under their care.
* **User Authentication System:**
    * **Secure Login Portal:** Validates user credentials against hashed passwords stored in the database.
    * **Full Password Reset Workflow:**
        1.  A "Forgot Password" link on the login page.
        2.  A UI to request a reset for a given username.
        3.  A secure, randomly generated temporary password is created and displayed (for demo purposes).
        4.  The system forces the user to set a new, permanent password upon their next login.
    * **Password Strength Enforcement:** A real-time checker ensures new passwords meet complexity requirements (length, uppercase, lowercase, numbers, special characters).
* **Security Monitoring & Brute-Force Protection:**
    * **Account Lockout:** The system automatically locks user accounts after 3 consecutive failed login attempts.
    * **Admin Security Dashboard:** A dedicated UI for administrators to view all locked accounts and recent failed login attempts.

---
## III. Analytics, Prediction & Reporting

A comprehensive suite of tools to transform raw data into strategic insights and predictive foresight.

* **Executive Dashboard:**
    * **KPI Metrics:** At-a-glance cards for Total Investment, Sessions Delivered, Individuals Served, Average Outcome Score, Cost Efficiency, and Quality Ratings.
    * **Cost vs. Effectiveness Quadrant:** An interactive Plotly scatter plot that visually segments interventions into four categories (e.g., "High Outcome, Low Cost") for quick strategic assessment.
* **Predictive Analytics Engine:**
    * **Model:** A powerful **XGBoost Classifier** for high performance and accuracy.
    * **Automated ML Pipeline:** A complete, end-to-end pipeline built with `scikit-learn` that automatically performs:
        1.  **Feature Engineering:** One-hot encodes categorical data (like intervention type, disability) to provide rich context.
        2.  **Feature Selection:** Uses `SelectKBest` to automatically identify and use only the most statistically relevant features, reducing noise and improving model stability.
        3.  **Hyperparameter Tuning:** Employs `GridSearchCV` to systematically test dozens of model configurations and select the optimal combination, maximizing performance and minimizing overfitting.
    * **Interactive Prediction Tool:**
        1.  An intuitive UI to input hypothetical session parameters.
        2.  Delivers an instant prediction of the outcome category (**Low, Medium, or High**).
        3.  Displays a bar chart showing the model's **confidence score** for each possible outcome.
* **Comprehensive Reporting Suite:**
    * **Multi-Format Export Engine:** Every report can be exported as:
        1.  A professionally formatted **PDF** document, including all text, tables, and visualizations.
        2.  A raw **CSV** data file for external analysis.
    * **Fully Implemented Report Templates:**
        * `Executive Summary`: High-level KPIs and strategic recommendations.
        * `Intervention Effectiveness`: Deep dive into cost-effectiveness, ROI, and statistical significance between programs.
        * `Cost Analysis`: Detailed breakdown of spending by category and program.
        * `Quality Assurance`: Focuses on attendance, session quality ratings, and flags sessions needing review.
        * `Individual Progress`: A detailed look at a single individual's journey.
        * `Staff Performance`: Compares staff metrics like session volume and average quality ratings.
        * `Regulatory Compliance`: Generates an auditable log of all services delivered, formatted for compliance checks.

---
## 🗂IV. Data & System Management

A full administrative backend for managing the application's data, configuration, and operational logic.

* **Data Quality Toolkit:**
    * **Issues Dashboard:** Summarizes all known data integrity problems.
    * **Interactive Correction Tools:**
        * **Duplicate Finder & Merger:** Automatically finds duplicate session records and allows an admin to merge them with a single click.
        * **Invalid Score Editor:** Lists all records with scores outside their defined range and allows for direct in-app editing and saving.
        * **Future-Date Corrector:** Lists all records dated in the future, allowing for bulk or individual correction.
* **Clinical Configuration Module:**
    * **Outcome Metrics Management:** A full CRUD (Create, Read, Update, Archive) interface for the metrics that define success.
    * **Assessment Scheduling System:** A UI to create recurring assessment schedules (Weekly, Monthly, etc.) for individuals, with automatically calculated "Next Due" dates.
    * **Clinical Protocol Manager:** A system to create and manage standardized protocols with rich text formatting via Markdown.
* **Financial Configuration Module:**
    * **Cost Category Manager:** Define overhead allocation percentages (e.g., Staff, Materials, Admin) for accurate financial modeling.
    * **Annual Budget Planner:** A tool to set the total budget for each fiscal year and view a real-time "Budget vs. Actuals" report with visualizations.
* **System Maintenance & Demo Toolkit:**
    * **"Reset Demo Data" Button:** A powerful feature for presentations. It completely wipes and re-seeds the database with a fresh, standardized sample dataset, which *includes* deliberate data errors to perfectly showcase the Data Quality Toolkit every time.

---
## V. User Interface & Experience

Features focused on making the application intuitive, responsive, and easy to use.

* **Clean, Tab-Based Navigation:** Information is logically grouped into expandable sections and tabs, preventing clutter.
* **Role-Aware UI:** The entire navigation sidebar dynamically changes based on the logged-in user's role, hiding irrelevant options.
* **Clear User Feedback:** The application provides constant feedback through spinners during long operations and clear success, warning, and error messages.
* **Dynamic Table Sizing:** Data tables automatically resize to fit their content, eliminating ugly scrollbars and ensuring all information is visible.
