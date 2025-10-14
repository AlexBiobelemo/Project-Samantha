# Codebase Analysis: Data Structures and Algorithms

## Executive Summary

This document provides a comprehensive analysis of the data structures, algorithms, and architectural patterns employed within the `app.py` codebase for Project Samantha, an Enterprise Care Analytics Platform. The analysis aims to offer insights into the application's design, performance characteristics, and areas for potential improvement.

**Key Findings:**

*   **Data Structures:** The application heavily relies on `pandas.DataFrame` for efficient tabular data manipulation and analysis, complemented by standard Python `list`, `dict`, and `set` for various data handling needs. Custom data structures are minimal, with a `PDF` class extending `fpdf2.FPDF` for custom report generation.
*   **Algorithms:** Core algorithms include robust password hashing (PBKDF2_HMAC), a comprehensive data cleaning and imputation pipeline, statistical significance and effect size calculations, and a sophisticated predictive modeling pipeline utilizing XGBoost for outcome classification. These algorithms are fundamental to the application's security, data quality, and analytical capabilities.
*   **Architectural Patterns:** Project Samantha is built as a **Monolithic Streamlit application** following a **Client-Server pattern**. It is highly **data-centric**, leveraging DuckDB as an embedded analytical database. While monolithic, the codebase demonstrates good internal modularity with distinct functions for UI, data access, security, and analytics.
*   **Performance Hotspots:** Potential performance bottlenecks include complex DuckDB queries (especially `get_comprehensive_outcome_data`), extensive Pandas DataFrame operations on large datasets, the computationally intensive `GridSearchCV` within the machine learning pipeline, and the image conversion process during PDF report generation. The lack of strategic Streamlit caching for expensive operations is a significant concern.
*   **Comparative Analysis:** The technology choices (DuckDB, PBKDF2_HMAC, Pandas, XGBoost, fpdf2) are generally well-justified for an embedded analytical application, balancing performance, security, and ease of development against more complex or resource-intensive alternatives.
*   **Dependencies:** The application has clear internal dependencies (e.g., UI functions calling data retrieval functions, authentication relying on hashing functions) and external dependencies on a rich ecosystem of Python libraries for data science, visualization, and security.
*   **Testing Coverage:** A critical observation is the **absence of automated testing**. The current reliance on manual testing introduces significant risks for regressions, hinders maintainability, and makes refactoring challenging. Key algorithms and database interactions are particularly vulnerable without a robust test suite.

**Overall Assessment & Recommendations:**

Project Samantha demonstrates a solid foundation for an analytical platform, with well-chosen data structures and algorithms for its core functions. However, its monolithic nature and the complete absence of automated testing pose significant challenges for future scalability, reliability, and maintainability.

**Key Recommendations:**
1.  **Implement Automated Testing:** Introduce a testing framework (e.g., `pytest`) and prioritize writing unit and integration tests for critical business logic, algorithms, and database interactions.
2.  **Strategic Caching:** Implement Streamlit's `@st.cache_data` and `@st.cache_resource` decorators for expensive data loading and processing functions (e.g., `get_comprehensive_outcome_data`, `get_advanced_analytics`, `perform_predictive_modeling`) to improve UI responsiveness.
3.  **Database Indexing:** Review DuckDB schema for potential indexing opportunities on frequently queried columns (beyond primary keys) to optimize query performance.
4.  **Modularization:** As the application grows, consider refactoring into a more modular structure (e.g., separate files for data access, analytics, UI components) to improve organization and maintainability.
5.  **Performance Optimization:** Investigate more vectorized Pandas operations where `df.apply()` is currently used, and explore more efficient PDF table rendering methods for large datasets.

## Data Structures Section

### Built-in/Standard Library Data Structures

This section documents the usage of standard Python data structures and those provided by common libraries within `app.py`.

#### 1. `list`
- **Usage:** Used extensively throughout the application for various purposes:
    - Storing collections of items (e.g., `users`, `intervention_data`, `outcome_metrics_data`, `age_groups`, `genders`, `disabilities`, `severities`, `support_levels`, `funding_sources` for sample data initialization).
    - Accumulating results (e.g., `user_ids`, `intervention_ids`, `metric_ids` during sample data creation).
    - Managing options for Streamlit select boxes and multiselects (e.g., `intervention_options`, `age_options`, `disability_options`, `selected_metrics`).
    - Storing errors and warnings during data validation (`errors`, `warnings` in `validate_bulk_outcome_data`).
    - Representing content for PDF reports (`content` list in `create_pdf_report` and report generation functions).
    - Storing security requirements (`missing_requirements` in `show_password_reset_form`).
- **Performance-critical usages:** While lists are fundamental, their use in large loops or for frequent lookups (without conversion to sets/dicts) could become a performance consideration. For example, `missing_individuals`, `missing_interventions`, `missing_metrics` are converted to sets for efficient lookup.

#### 2. `dict` (Dictionary)
- **Usage:** Used for key-value pair storage, often representing structured data or configurations:
    - Storing user session state (`st.session_state`).
    - Passing arguments to functions (e.g., `filters` in `get_comprehensive_outcome_data`, `analytics` results).
    - Representing individual user data (`user` in `authenticate_user`).
    - Storing password strength requirements (`requirements` in `check_password_strength`).
    - Storing content for PDF reports (`item` in `create_pdf_report`, `kpi_data` in `generate_executive_report`).
    - Mapping filter keys to database columns (`filter_map` in `get_comprehensive_outcome_data`).
    - Storing quality metrics (`quality_metrics` in `get_advanced_analytics`).
    - Storing model results (`model_results` in `perform_predictive_modeling`).
    - Storing system settings (`current_settings`, `new_settings` in `show_financial_settings`).
    - Storing permissions (`permissions` in `show_advanced_user_management`).
    - Storing record counts (`record_counts` in `show_system_maintenance`).
- **Performance-critical usages:** Dictionaries provide O(1) average time complexity for lookups, insertions, and deletions, making them efficient for managing dynamic, associative data.

#### 3. `set`
- **Usage:** Used for storing unique, unordered collections of items, primarily for efficient membership testing.
    - In `validate_bulk_outcome_data`, `existing_individual_ids`, `existing_intervention_names`, `existing_metric_names` are created as sets to quickly check if an ID/name exists in the database (O(1) average time complexity for `in` operator).
    - Used to find missing references by set difference (`missing_individuals`, `missing_interventions`, `missing_metrics`).
- **Performance-critical usages:** Crucial for optimizing validation checks against large lists of existing entities, preventing O(N) lookups in lists.

#### 4. `tuple`
- **Usage:** Used for immutable sequences of items, often for fixed-size collections or return values from functions.
    - Date ranges are often passed as tuples (e.g., `date_range` in `get_comprehensive_outcome_data`).
    - Return values from database queries (`conn.execute(...).fetchone()`).
    - Return values from functions like `hash_password` (`Tuple[str, str]`).
- **Performance-critical usages:** Immutability can offer minor performance benefits and ensures data integrity for fixed collections.

#### 5. `pandas.DataFrame`
- **Usage:** The primary data structure for tabular data manipulation and analysis. Used extensively for:
    - Storing results from DuckDB queries (`conn.execute(...).df()`).
    - Performing data cleaning, imputation, and transformation (`get_comprehensive_outcome_data`).
    - Aggregating data for analytics (`groupby` operations in `get_advanced_analytics`).
    - Displaying data in Streamlit tables (`st.dataframe`).
    - Feature engineering and preparation for machine learning (`perform_predictive_modeling`).
    - Storing and manipulating data for report generation (`generate_*_report` functions).
- **Performance-critical usages:** Pandas DataFrames are optimized for vectorized operations, making them highly efficient for large-scale data processing compared to manual Python loops. Operations like `groupby`, `apply`, and arithmetic operations on columns are heavily optimized.

#### 6. `numpy.ndarray`
- **Usage:** Used for numerical operations, especially within statistical and machine learning contexts.
    - `np.sqrt`, `np.var`, `np.mean` in `calculate_effect_size`.
    - Implicitly used by pandas DataFrames for numerical columns.
- **Performance-critical usages:** NumPy arrays provide efficient storage and operations for numerical data, which is critical for the performance of statistical calculations and machine learning algorithms.

#### 7. `datetime.datetime`, `datetime.date`, `datetime.timedelta`
- **Usage:** Used for handling dates and times throughout the application:
    - Storing timestamps in the database (e.g., `created_at`, `updated_at`, `last_login`, `password_expires`).
    - Calculating date ranges for filtering data (`date_range` in `get_comprehensive_outcome_data`).
    - Determining password expiration (`password_expires` in `authenticate_user`).
    - Generating realistic session dates in sample data (`_generate_realistic_outcome_data`).
    - Calculating age of data (`days_since_activity` in `show_system_maintenance`).
- **Performance-critical usages:** Essential for time-series analysis and date-based filtering, which are common in analytics applications.

#### 8. `hashlib`
- **Usage:** Used for cryptographic hashing, specifically for password storage.
    - `hashlib.sha256` in `hash_password` and `verify_password` (legacy).
    - `hashlib.pbkdf2_hmac` in `hash_password` (enhanced security).
- **Performance-critical usages:** Security-critical for protecting user credentials. The choice of PBKDF2_HMAC with a high iteration count (`100000`) is a deliberate performance trade-off for increased security against brute-force attacks.

#### 9. `uuid`
- **Usage:** Used for generating universally unique identifiers (UUIDs).
    - Generating primary keys for all database tables (e.g., `id` for `users`, `facilities`, `interventions`, etc.).
    - Generating session IDs (`session_id`) and record IDs (`record_id`) for outcome and cost records.
    - Generating salts for password hashing (`uuid.uuid4().bytes`).
- **Performance-critical usages:** Ensures uniqueness across distributed systems and prevents ID collisions, which is crucial for data integrity.

#### 10. `io`
- **Usage:** Used for in-memory I/O operations, particularly for handling binary data.
    - `io.BytesIO` for converting Plotly figures to in-memory PNG images before embedding them in PDF reports (`create_pdf_report`).
- **Performance-critical usages:** Avoids writing temporary files to disk, improving performance and reducing disk I/O for report generation.

#### 11. `json`
- **Usage:** Used for serializing and deserializing Python objects to/from JSON format.
    - Storing `old_values` and `new_values` in the `audit_log` table as JSON strings.
- **Performance-critical usages:** Standard format for structured data exchange and storage.

#### 12. `base64`
- **Usage:** Used for encoding and decoding binary data to/from base64 strings.
    - Encoding password hashes and salts in `hash_password` and `verify_password`.
- **Performance-critical usages:** Ensures binary data (like hashes) can be safely stored and transmitted in text-based systems.

#### 13. `typing` (Dict, List, Optional, Tuple)
- **Usage:** Used for type hinting, improving code readability, maintainability, and enabling static analysis.
    - `Dict`, `List`, `Optional`, `Tuple` are used to specify expected types for function parameters and return values.
- **Performance-critical usages:** Not directly performance-critical at runtime, but crucial for developer productivity and preventing type-related bugs, especially in larger codebases.

#### 14. `re` (Regular Expressions)
- **Usage:** Used for pattern matching in strings.
    - `re.search` in `check_password_strength` to validate password complexity (uppercase, lowercase, digit, special character).
- **Performance-critical usages:** Efficient for complex string pattern matching.

#### 15. `scipy.stats`
- **Usage:** Used for statistical functions.
    - `stats.ttest_ind` for independent t-tests in `calculate_statistical_significance`.
    - `stats.mannwhitneyu` for Mann-Whitney U test in `calculate_statistical_significance`.
    - `stats.linregress` for linear regression in `show_individual_detailed_analytics`.
- **Performance-critical usages:** Provides optimized implementations of statistical tests.

#### 16. `sklearn` (scikit-learn)
- **Usage:** Used for machine learning tasks.
    - `sklearn.ensemble.RandomForestRegressor` (imported but not used in the provided code snippet).
    - `sklearn.model_selection.train_test_split` for splitting data into training and testing sets.
    - `sklearn.preprocessing.StandardScaler` for feature scaling.
    - `sklearn.pipeline.Pipeline` for chaining multiple processing steps.
    - `sklearn.model_selection.GridSearchCV` for hyperparameter tuning.
    - `sklearn.feature_selection.SelectKBest`, `f_classif` for feature selection.
    - `sklearn.metrics.accuracy_score` (imported but not explicitly used for reporting in the provided code, `best_model.score` is used).
    - `sklearn.preprocessing.LabelEncoder` for encoding categorical target variables.
- **Performance-critical usages:** Provides highly optimized and robust implementations of machine learning algorithms and utilities.

#### 17. `fpdf.FPDF` (from `fpdf2`)
- **Usage:** Used for generating PDF documents.
    - The `PDF` class inherits from `FPDF` to customize headers and footers.
    - `create_pdf_report` uses `PDF` instance methods (`add_page`, `set_font`, `cell`, `multi_cell`, `image`, `output`) to construct the PDF.
- **Performance-critical usages:** Enables programmatic PDF generation for reporting.

#### 18. `xgboost.XGBRegressor`, `xgboost.XGBClassifier`
- **Usage:** Used for gradient boosting machine learning models.
    - `XGBClassifier` is used in `perform_predictive_modeling` for classification tasks.
    - `XGBRegressor` is imported but not used in the provided code.
- **Performance-critical usages:** XGBoost is known for its high performance and accuracy in machine learning competitions and real-world applications.

#### 19. `duckdb`
- **Usage:** Used as an in-process SQL OLAP database.
    - `duckdb.connect` to establish a connection to `samantha_data.db`.
    - `conn.execute` for all SQL queries (CREATE TABLE, INSERT, UPDATE, SELECT, DELETE).
    - `conn.execute(...).df()` to fetch query results directly into a pandas DataFrame.
- **Performance-critical usages:** DuckDB is designed for analytical queries on large datasets, offering high performance for data retrieval and aggregation, which is central to the application's analytics capabilities.

### Custom Data Structures

#### 1. `class PDF(FPDF)`
- **Name and location:** `PDF` class, defined in `app.py` (lines 27-35).
- **Purpose and intended use cases:** This class extends `FPDF` to provide custom header and footer functionality for PDF reports generated by the application. It's used specifically by the `create_pdf_report` function to ensure consistent branding and page numbering across all generated PDF reports.
- **Why this custom implementation was necessary:** To encapsulate the specific header and footer layout required for "Project Samantha" reports, rather than re-implementing them for every PDF generation.
- **Technical Details:**
    - **Internal representation:** Inherits all internal state and mechanisms from the `FPDF` class, which manages the PDF document structure, fonts, pages, and content streams.
    - **Key properties and invariants maintained:** Ensures that every page of the PDF includes the specified header ("Project Samantha - Comprehensive Analytics Report") and footer (page number).
    - **Memory layout and allocation strategy:** Managed by the underlying `FPDF` library.
- **Operations Analysis:**
    - `header()`: Called automatically by `FPDF` when a new page is added. Sets font, adds a cell for the title, and adds a line break.
    - `footer()`: Called automatically by `FPDF` at the bottom of each page. Sets font, adds a cell for the page number.
- **Dependencies:** Depends on the `FPDF` class from the `fpdf` library.
- **Usage Patterns:** Instantiated once per PDF report generation within `create_pdf_report`.
- **Code Quality:** Simple, clear, and directly extends the base class as intended. No explicit thread safety concerns as `FPDF` objects are typically used in a single-threaded context for document generation. Error handling is implicitly managed by `FPDF` for PDF generation issues.

## Algorithms Section

This section documents the significant algorithms implemented or utilized within `app.py`.

## Architectural Patterns

The `app.py` application primarily follows a **Monolithic Architecture** with a **Client-Server (Streamlit) pattern**. While the entire application resides in a single file, it exhibits internal modularity through well-defined functions and classes, each handling specific concerns.

### 1. Monolithic Architecture
- **Description:** The entire application, encompassing the user interface, business logic, data access, and even database initialization, is contained within a single `app.py` file.
- **Rationale:** This approach is common for smaller Streamlit applications, offering simplicity in deployment and development.
- **Implications:** While easy to start, a monolithic structure can become challenging to scale and maintain as the application grows in complexity.

### 2. Client-Server Pattern (Streamlit Framework)
- **Description:** Streamlit inherently operates on a client-server model. The `app.py` script executes on a server, and the generated interactive user interface is rendered in the user's web browser (client). Streamlit manages the communication and state synchronization between the client and server.
- **Components:**
    - **Client:** Web browser displaying the Streamlit UI.
    - **Server:** Python process running `app.py`, handling data processing, logic execution, and UI updates.

### 3. Data-Centric Architecture
- **Description:** The application's core functionality revolves heavily around data management, processing, and analysis.
- **Key Technologies:**
    - **DuckDB:** Used as an embedded, in-process analytical database for efficient SQL queries and data storage (`samantha_data.db`).
    - **Pandas DataFrames:** Central to data manipulation, cleaning, imputation, aggregation, and preparation for analytics and machine learning. Vectorized operations are heavily leveraged for performance.

### 4. Modular Design (within the monolith)
- **Description:** Despite being a single file, the codebase is structured into distinct, cohesive functions and classes, promoting a separation of concerns.
- **Examples:**
    - **Database Layer:** `init_database`, `get_comprehensive_outcome_data`, `create_audit_log`.
    - **Authentication & Security:** `authenticate_user`, `hash_password`, `check_password_strength`.
    - **UI Components:** Functions prefixed with `show_` (e.g., `show_executive_dashboard`, `show_smart_data_entry`).
    - **Reporting:** `create_pdf_report`, `generate_executive_report`.
    - **Analytics:** `get_advanced_analytics`, `calculate_statistical_significance`, `perform_predictive_modeling`.

### 5. Session State Management
- **Description:** Streamlit's `st.session_state` is extensively used to persist data and maintain the application's state across user interactions and reruns.
- **Usage:** Stores authentication status (`st.session_state.authenticated`), user information (`st.session_state.user`), and the database connection object (`st.session_state.db_conn`).

### 6. Reporting and Visualization Layer
- **Description:** Dedicated components for generating interactive data visualizations and static PDF reports.
- **Key Technologies:**
    - **Plotly Express/Graph Objects:** Used for creating a wide array of interactive charts and graphs displayed within the Streamlit UI and embedded in PDF reports.
    - **fpdf2:** Utilized for programmatic generation of multi-page PDF reports, including custom headers and footers.

### 7. Security Layer
- **Description:** A set of functions dedicated to user authentication, password management, and activity logging to ensure data security and accountability.
- **Components:**
    - Password Hashing (`hash_password`): Uses PBKDF2_HMAC for secure password storage.
    - Password Verification (`verify_password`): Authenticates users against stored hashes.
    - Password Strength Check (`check_password_strength`): Enforces strong password policies.
    - Audit Logging (`create_audit_log`): Records significant data modifications.
    - Alert Generation (`generate_alert`): Notifies administrators of important system events.

### 8. Machine Learning Pipeline
- **Description:** The `perform_predictive_modeling` function encapsulates a complete machine learning workflow for predictive analytics.
- **Components:**
    - Data Preparation (Pandas, LabelEncoder, One-Hot Encoding).
    - Data Splitting (`train_test_split`).
    - Feature Selection (`SelectKBest`).
    - Feature Scaling (`StandardScaler`).
    - Model Training (`XGBClassifier`).
    - Hyperparameter Tuning (`GridSearchCV`).

This architectural overview highlights the application's structure and the patterns employed to manage its various functionalities, from data handling to user interaction and advanced analytics.

## Performance Analysis Section

This section identifies potential performance hotspots within the `app.py` codebase, focusing on areas that could become bottlenecks as data volume or user load increases.

### 1. Database Queries (DuckDB)
- **Hotspot:** Functions that perform complex or frequent database queries, particularly `get_comprehensive_outcome_data`.
- **Details:**
    - `get_comprehensive_outcome_data`: This function executes a SQL query involving multiple `JOIN` operations and conditional `WHERE` clauses. As the `outcome_records` table grows, the complexity of this query could lead to increased execution times. While DuckDB is optimized for analytical queries, the lack of explicit indexing in the `CREATE TABLE` statements (beyond primary keys) might impact performance on non-PK `WHERE` clauses or `JOIN` conditions.
    - **Repeated Small Queries:** Although DuckDB is fast, frequent individual `SELECT` statements within loops (e.g., fetching `latest_score` in `show_individual_goal_tracking` or during `_generate_realistic_outcome_data` for baseline scores) can accumulate overhead. Batching these operations or pre-fetching data where possible could improve efficiency.

### 2. Pandas DataFrame Operations
- **Hotspot:** Extensive DataFrame manipulations, especially on large datasets.
- **Details:**
    - `get_comprehensive_outcome_data` and `get_advanced_analytics`: These functions involve numerous Pandas operations such as filtering, grouping (`groupby`), aggregation (`agg`), and column-wise calculations. While Pandas is highly optimized for vectorized operations, processing extremely large DataFrames can still be memory-intensive and time-consuming.
    - `df.apply()`: The use of `df.apply()` with `lambda` functions (e.g., for `normalized_score` and `improvement_from_baseline` calculation in `get_comprehensive_outcome_data`) can be slower than fully vectorized Pandas or NumPy operations, especially for large numbers of rows.

### 3. Machine Learning Pipeline (`perform_predictive_modeling`)
- **Hotspot:** The entire machine learning model training and hyperparameter tuning process.
- **Details:**
    - `GridSearchCV`: This component is inherently computationally expensive. It trains multiple models across different hyperparameter combinations and cross-validation folds. With a complex model like `XGBClassifier`, this can consume significant CPU resources and time. Although `n_jobs=-1` is used for parallelization, it's still a major bottleneck for real-time or frequent model retraining.
    - **One-Hot Encoding (`pd.get_dummies`):** If the categorical features have a high cardinality (many unique values), one-hot encoding can generate a very wide DataFrame, leading to increased memory consumption and slower subsequent processing steps.

### 4. PDF Report Generation (`create_pdf_report`)
- **Hotspot:** Conversion of Plotly figures to images and iterative table rendering.
- **Details:**
    - `data.to_image()`: Converting interactive Plotly figures into static PNG images for embedding in PDFs can be CPU and memory intensive, especially if high resolution or many figures are involved. This operation requires an external renderer (like `kaleido`), which adds overhead.
    - **Iterative Table Rendering:** The current implementation of table rendering in `create_pdf_report` iterates row by row and cell by cell. For DataFrames with many rows and columns, this can be significantly slower than using more optimized table rendering features provided by `fpdf2` or other libraries.

### 5. Streamlit Reruns and Caching
- **Hotspot:** Unnecessary re-execution of expensive functions on every Streamlit rerun.
- **Details:** Streamlit applications rerun the entire script from top to bottom on almost every user interaction. If computationally intensive functions (like `get_comprehensive_outcome_data`, `get_advanced_analytics`, or `perform_predictive_modeling`) are not properly cached using `@st.cache_data` or `@st.cache_resource`, they will be re-executed repeatedly, leading to a sluggish user experience and high resource consumption. Currently, `get_comprehensive_outcome_data` is called without caching in several places, which is a significant potential hotspot.

Addressing these areas through optimization techniques (e.g., database indexing, more vectorized Pandas operations, strategic caching, and potentially asynchronous processing for ML tasks) would significantly improve the application's scalability and responsiveness.

## Comparative Analysis

This section provides a comparative analysis of key data structures, algorithms, and technologies used in `app.py` against common alternatives, highlighting the rationale behind the current choices and their trade-offs.

### 1. Database Choice: DuckDB vs. Alternatives

-   **Current Choice:** DuckDB (in-process, analytical SQL database).
-   **Alternatives:**
    -   **Traditional RDBMS (e.g., PostgreSQL, MySQL, SQLite):**
        -   **Pros:** Mature, robust, widely supported, good for OLTP (Online Transactional Processing), strong concurrency.
        -   **Cons:** Can be overkill for embedded analytics, often requires separate server processes, may be slower for complex analytical queries on large datasets compared to OLAP-optimized solutions. SQLite is simpler but less performant for complex analytics.
    -   **NoSQL Databases (e.g., MongoDB, Cassandra):**
        -   **Pros:** Flexible schema, horizontal scalability, good for unstructured/semi-structured data.
        -   **Cons:** Not ideal for complex relational queries, lacks strong ACID guarantees (often), steeper learning curve for SQL users.
-   **Justification for DuckDB:**
    -   **Performance for Analytics:** DuckDB is specifically designed for OLAP workloads, offering superior performance for complex analytical queries (joins, aggregations) on large datasets compared to general-purpose RDBMS or SQLite.
    -   **Embedded Nature:** Being an in-process database, it simplifies deployment and management, as it doesn't require a separate database server. This is ideal for a self-contained Streamlit application.
    -   **SQL Compatibility:** Provides a familiar SQL interface, reducing the learning curve for developers already proficient in SQL.
    -   **Trade-offs:** Not designed for high-concurrency OLTP workloads or distributed systems. The single-file database (`samantha_data.db`) can become large.

### 2. Password Hashing Algorithm: PBKDF2_HMAC vs. Alternatives

-   **Current Choice:** PBKDF2_HMAC with SHA256 and 100,000 iterations.
-   **Alternatives:**
    -   **Direct Hashing (e.g., `hashlib.sha256` without salt/iterations):**
        -   **Pros:** Fast.
        -   **Cons:** Extremely insecure. Vulnerable to rainbow table attacks and fast brute-force attacks. (Used as a legacy fallback in `app.py`, but not for new hashes).
    -   **Bcrypt:**
        -   **Pros:** Industry-standard, designed to be slow and adaptive (cost factor can be increased over time), includes salt.
        -   **Cons:** Not natively in Python's `hashlib`, requires external libraries (`bcrypt`).
    -   **Scrypt:**
        -   **Pros:** Designed to be memory-hard (resists GPU/ASIC attacks), includes salt.
        -   **Cons:** Not natively in Python's `hashlib`, requires external libraries (`pyscript`).
    -   **Argon2:**
        -   **Pros:** Winner of the Password Hashing Competition, highly configurable for memory, time, and parallelism. Considered the strongest modern algorithm.
        -   **Cons:** Not natively in Python's `hashlib`, requires external libraries (`argon2-cffi`).
-   **Justification for PBKDF2_HMAC:**
    -   **Security Improvement:** A significant improvement over direct SHA256 hashing by incorporating a salt and a high iteration count, making brute-force attacks computationally expensive.
    -   **Standard Library Availability:** Available directly in Python's `hashlib` module, avoiding external dependencies for this critical security function.
    -   **Trade-offs:** While more secure than simple hashes, it is generally considered less resistant to specialized hardware attacks (ASICs/GPUs) compared to memory-hard algorithms like Scrypt or Argon2. For a new, high-security application, Argon2 would be the preferred choice.

### 3. Data Manipulation: Pandas DataFrames vs. Raw Python Structures

-   **Current Choice:** `pandas.DataFrame` for tabular data.
-   **Alternatives:**
    -   **Raw Python Lists of Dictionaries/Tuples:**
        -   **Pros:** Native to Python, no external dependency.
        -   **Cons:** Slower for large-scale numerical and string operations due to Python's interpreted nature and lack of vectorized operations. Requires manual looping, which is less efficient and more verbose for data analysis tasks.
-   **Justification for Pandas DataFrames:**
    -   **Vectorized Operations:** Pandas leverages NumPy arrays and C-optimized routines, enabling highly efficient, vectorized operations on entire columns or DataFrames. This drastically outperforms explicit Python loops for data cleaning, transformation, and aggregation.
    -   **Ease of Use:** Provides a rich API for common data manipulation tasks, making data analysis code more concise, readable, and less error-prone.
    -   **Integration:** Seamlessly integrates with other scientific computing libraries like NumPy, SciPy, and Scikit-learn.
    -   **Trade-offs:** Introduces an external dependency and a slight learning curve for those unfamiliar with its API. Can be memory-intensive for extremely large datasets that don't fit into RAM.

### 4. Machine Learning Model: XGBoost Classifier vs. Alternatives

-   **Current Choice:** `xgboost.XGBClassifier` within a `sklearn.pipeline.Pipeline`.
-   **Alternatives:**
    -   **Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`):**
        -   **Pros:** Robust, handles non-linear relationships, less prone to overfitting than single decision trees, good for mixed data types.
        -   **Cons:** Can be slower to train than gradient boosting for very large datasets, less interpretable than simpler models.
    -   **Logistic Regression (`sklearn.linear_model.LogisticRegression`):**
        -   **Pros:** Simple, fast to train, highly interpretable, good baseline model.
        -   **Cons:** Assumes linearity, may not capture complex relationships in data, can be sensitive to feature scaling.
    -   **Support Vector Machines (SVMs) (`sklearn.svm.SVC`):**
        -   **Pros:** Effective in high-dimensional spaces, memory efficient (subset of training points), versatile with different kernels.
        -   **Cons:** Can be computationally expensive for large datasets, sensitive to feature scaling, less interpretable.
-   **Justification for XGBoost Classifier:**
    -   **High Performance and Accuracy:** XGBoost is a highly optimized gradient boosting library known for its speed and state-of-the-art predictive accuracy in many machine learning competitions and real-world applications.
    -   **Handles Mixed Data Types:** Effectively deals with both numerical and categorical features (after appropriate encoding).
    -   **Robustness:** Less prone to overfitting compared to simpler boosting methods, and includes regularization.
    -   **Feature Importance:** Provides clear feature importance scores, aiding in understanding model decisions.
    -   **Trade-offs:** Can be computationally intensive to train, especially with hyperparameter tuning (`GridSearchCV`). Requires careful tuning to prevent overfitting.

### 5. PDF Generation Library: fpdf2 vs. Alternatives

-   **Current Choice:** `fpdf2` (a fork of FPDF).
-   **Alternatives:**
    -   **ReportLab:**
        -   **Pros:** Very powerful, highly flexible, suitable for complex, programmatic PDF generation.
        -   **Cons:** Steeper learning curve, more verbose API, can be overkill for simpler reporting needs.
    -   **WeasyPrint:**
        -   **Pros:** Renders HTML and CSS to PDF, allowing web developers to use familiar tools.
        -   **Cons:** Requires a full HTML/CSS rendering engine, potentially heavier dependency.
    -   **PyPDF2 / PyMuPDF:**
        -   **Pros:** Primarily for manipulating existing PDFs (splitting, merging, extracting), not for generating from scratch.
-   **Justification for fpdf2:**
    -   **Simplicity and Direct Control:** Offers a straightforward, imperative API for building PDFs page by page, which is suitable for generating structured reports with text, tables, and images.
    -   **Lightweight:** Relatively lightweight dependency compared to solutions requiring a full rendering engine.
    -   **Customization:** Easy to extend (as seen with the `PDF` class) for custom headers, footers, and layouts.
    -   **Trade-offs:** Requires manual positioning and layout management, which can be tedious for highly dynamic or complex layouts compared to HTML-to-PDF renderers.

This comparative analysis demonstrates that the choices made in `app.py` generally align with the application's analytical and reporting needs, balancing performance, security, and ease of development, while acknowledging areas where more advanced alternatives exist for specific trade-offs.

## Dependency Graph Section

This section maps the key relationships and dependencies between different components within the `app.py` application, including database schema, function call hierarchies, and external library integrations.

### 1. Database Schema Dependencies

The application's data model is built around a relational schema managed by DuckDB. The following outlines the primary tables and their relationships:

-   **`users`**: Stores user authentication and profile information.
    -   `created_by` in `interventions`, `outcome_metrics`, `individuals`, `treatment_goals`, `system_settings`, `budgets`, `clinical_protocols`, `assessment_schedules` refers to `users.id`.
    -   `recorded_by`, `staff_id`, `verified_by` in `outcome_records` refer to `users.id`.
    -   `changed_by` in `audit_log` refers to `users.id`.
    -   `updated_by` in `system_settings` refers to `users.id`.
    -   `generated_by` in `report_cache` refers to `users.id`.
    -   `target_user` in `alerts` refers to `users.id`.
-   **`facilities`**: Stores facility-specific settings and details.
    -   `facility_id` in `interventions`, `outcome_metrics`, `individuals` refers to `facilities.id`.
-   **`interventions`**: Defines various care interventions.
    -   `intervention_id` in `outcome_records`, `cost_records` refers to `interventions.id`.
-   **`outcome_metrics`**: Defines the metrics used to measure outcomes.
    -   `outcome_metric_id` in `outcome_records`, `treatment_goals`, `assessment_schedules` refers to `outcome_metrics.id`.
-   **`individuals`**: Stores demographic and profile information for individuals receiving care.
    -   `individual_id` in `outcome_records`, `cost_records`, `treatment_goals`, `assessment_schedules` refers to `individuals.id`.
-   **`outcome_records`**: Stores individual session outcome data.
    -   Linked to `individuals`, `interventions`, `outcome_metrics`, `users` (staff_id, recorded_by).
    -   `session_id` links to `cost_records`.
-   **`cost_records`**: Stores cost data associated with interventions and sessions.
    -   Linked to `interventions`, `individuals`, `outcome_records` (via `session_id`).
-   **`treatment_goals`**: Stores individual treatment goals.
    -   Linked to `individuals`, `outcome_metrics`, `users`.
-   **`audit_log`**: Records changes to data for security and compliance.
    -   `changed_by` refers to `users.id`.
-   **`system_settings`**: Stores application-wide configuration.
    -   `updated_by` refers to `users.id`.
-   **`alerts`**: Stores system notifications.
    -   `target_user` refers to `users.id`.
-   **`assessment_schedules`**: Stores recurring assessment schedules.
    -   Linked to `individuals`, `outcome_metrics`, `users`.
-   **`clinical_protocols`**: Stores defined clinical protocols.
    -   `created_by` refers to `users.id`.
-   **`budgets`**: Stores annual budget information.
    -   `created_by` refers to `users.id`.

### 2. Function Call Dependencies (High-Level)

This outlines the call hierarchy and interdependencies between major functions:

-   **`main()`**:
    -   Initializes `st.session_state.db_conn` by calling `init_database()`.
    -   Manages authentication flow: calls `show_password_reset_form()`, `show_forgot_password_form()`, `login_form()`.
    -   Routes to various UI functions based on user selection: `show_executive_dashboard()`, `show_comprehensive_reporting()`, `show_individual_analytics()`, `show_advanced_intervention_analysis()`, `show_advanced_data_management()`, `show_comprehensive_system_config()`, `show_advanced_user_management()`, `show_staff_dashboard()`, `show_smart_data_entry()`.
    -   Calls `show_system_alerts()` to display notifications.

-   **Data Retrieval Functions (e.g., `get_comprehensive_outcome_data`, `get_interventions`, `get_outcome_metrics`, `get_individuals`, `get_facility_data`)**:
    -   All depend on `st.session_state.db_conn` to execute DuckDB queries.
    -   `get_comprehensive_outcome_data` is a critical data pipeline, performing cleaning, imputation, and derived metric calculation using `pandas` and `numpy`.

-   **Analytics Functions (e.g., `get_advanced_analytics`, `calculate_statistical_significance`, `calculate_effect_size`, `perform_predictive_modeling`)**:
    -   `get_advanced_analytics` heavily relies on `get_comprehensive_outcome_data` for its input data. It then uses `pandas` for aggregations and calls `calculate_statistical_significance` and `calculate_effect_size`.
    -   `perform_predictive_modeling` also depends on `get_comprehensive_outcome_data` and utilizes `pandas`, `sklearn`, and `xgboost` for its ML pipeline.

-   **Security Functions (e.g., `authenticate_user`, `hash_password`, `verify_password`, `check_password_strength`, `create_audit_log`, `generate_alert`)**:
    -   `authenticate_user` depends on `verify_password` for password validation and interacts with the `users` table.
    -   `verify_password` depends on `hash_password` for re-hashing the input password.
    -   `create_audit_log` and `generate_alert` interact directly with their respective database tables and are called by various UI/logic functions when significant events occur (e.g., data modification, system issues).

-   **Reporting Functions (e.g., `create_pdf_report`, `generate_executive_report`, etc.)**:
    -   `create_pdf_report` depends on `fpdf2` and `io` for PDF generation and embedding Plotly figures.
    -   `generate_*_report` functions depend on `get_comprehensive_outcome_data` and `get_advanced_analytics` for data, and `plotly.express`/`plotly.graph_objects` for visualizations.

### 3. Streamlit Session State Dependencies

`st.session_state` acts as a central hub for managing global application state and dependencies:

-   `st.session_state.db_conn`: Holds the active DuckDB connection, making it accessible globally to all functions interacting with the database.
-   `st.session_state.authenticated`: Boolean flag indicating user login status.
-   `st.session_state.user`: Dictionary containing details of the currently logged-in user (ID, username, role, etc.). This dictates access control and personalized content.
-   `st.session_state.password_reset_required`, `st.session_state.password_reset_flow`, `st.session_state.reset_username`: Flags and data for managing the password reset workflow.
-   `st.session_state.quick_entry_mode`: Flag for displaying the emergency data entry modal.

### 4. External Library Dependencies

The application relies on several external Python libraries, each contributing specific functionalities:

-   **`streamlit`**: Core framework for building the web application UI.
-   **`pandas`**: Essential for data manipulation, cleaning, and analysis (DataFrames).
-   **`duckdb`**: Embedded analytical database for data storage and querying.
-   **`plotly.express`, `plotly.graph_objects`**: For interactive data visualizations.
-   **`numpy`**: Numerical operations, especially within Pandas and statistical functions.
-   **`datetime`**: Date and time handling.
-   **`hashlib`, `uuid`, `base64`, `re`**: Security-related functions (hashing, UUID generation, regex).
-   **`scipy.stats`**: Statistical tests (t-test, Mann-Whitney U, linear regression).
-   **`sklearn` (scikit-learn)**: Machine learning utilities (train/test split, scaling, feature selection, pipelines, label encoding).
-   **`xgboost`**: Gradient Boosting machine learning models (XGBClassifier).
-   **`fpdf` (fpdf2)**: PDF document generation.
-   **`io`**: In-memory I/O for handling binary data (e.g., images for PDF).
-   **`json`**: JSON serialization/deserialization for audit logs.
-   **`typing`**: For type hinting, improving code readability and maintainability.

This section provides a clear overview of how different parts of the application are interconnected, which is crucial for understanding its overall structure and for future development or debugging efforts.

## Code Examples

This section provides concise code examples to illustrate the implementation and usage of key data structures and algorithms discussed in `app.py`.

### 1. Password Hashing and Verification

This example demonstrates the secure password hashing and verification process using PBKDF2_HMAC.

```python
import hashlib
import uuid
import base64
from typing import Tuple

def hash_password(password: str, salt: str = None) -> Tuple[str, str]:
    """Enhanced password hashing with salt"""
    if salt is None:
        salt = base64.b64encode(uuid.uuid4().bytes).decode()

    # Use PBKDF2 for better security
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return base64.b64encode(password_hash).decode(), salt

def verify_password(password: str, password_hash: str, salt: str = None) -> bool:
    """Verify password with enhanced security"""
    if salt:
        computed_hash, _ = hash_password(password, salt)
        return computed_hash == password_hash
    else:
        # Fallback for legacy hashes (less secure, for backward compatibility only)
        return hashlib.sha256(password.encode()).hexdigest() == password_hash

# Example Usage:
# new_hash, new_salt = hash_password("MySecurePassword123!")
# print(f"Hashed Password: {new_hash}")
# print(f"Salt: {new_salt}")
# is_correct = verify_password("MySecurePassword123!", new_hash, new_salt)
# print(f"Password correct: {is_correct}")
```

### 2. Data Cleaning and Imputation (from `get_comprehensive_outcome_data`)

This snippet illustrates the core Pandas operations for type conversion and imputation within the data pipeline.

```python
import pandas as pd
import numpy as np

# Assume df is a DataFrame loaded from DuckDB
# df = conn.execute(query, params).df()

if not df.empty:
    # STEP 1: Convert to numeric, coercing errors to NaN
    numeric_cols = [
        'score', 'baseline_score', 'target_score', 'cost_per_session', 'scale_min', 'scale_max',
        'direct_cost', 'indirect_cost', 'overhead_cost', 'material_cost', 'planned_duration', 'session_duration', 'quality_rating'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # STEP 2: Impute all NaNs
    numeric_cols_to_impute = ['cost_per_session', 'session_duration', 'quality_rating', 'direct_cost', 'indirect_cost', 'overhead_cost', 'material_cost', 'score', 'baseline_score']
    categorical_cols_to_impute = ['intervention_category', 'metric_category', 'age_group', 'disability_category', 'support_level', 'staff_name']
    
    for col in numeric_cols_to_impute:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
            
    for col in categorical_cols_to_impute:
        if col in df.columns:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col].fillna(mode_val[0], inplace=True)
            else:
                df[col].fillna("Unknown", inplace=True)

    # STEP 3: Calculate derived metrics (simplified for example)
    cost_cols = ['direct_cost', 'indirect_cost', 'overhead_cost', 'material_cost']
    df['total_cost'] = df[cost_cols].sum(axis=1)
    # ... further derived metrics ...
```

### 3. Predictive Modeling Pipeline Setup

This example shows the construction of the Scikit-learn pipeline with feature selection, scaling, and the XGBoost classifier, along with hyperparameter tuning using `GridSearchCV`.

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier

def setup_predictive_pipeline(X, y):
    """Sets up and tunes a predictive modeling pipeline."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Create a Pipeline
    pipeline = Pipeline([
        ('selector', SelectKBest(score_func=f_classif)),
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
    ])
    
    # Define a parameter grid to search over
    param_grid = {
        'selector__k': [20, 30],
        'model__n_estimators': [100, 150],
        'model__max_depth': [3, 4],
        'model__learning_rate': [0.05, 0.1]
    }
    
    # Set up and run the grid search
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    return best_model, X_train, y_train, X_test, y_test

# Example Usage (assuming X and y are prepared DataFrames/arrays):
# X_features = ... # DataFrame of features
# y_target = ...   # Series/array of target labels (encoded)
# best_model, X_train, y_train, X_test, y_test = setup_predictive_pipeline(X_features, y_target)
# train_accuracy = best_model.score(X_train, y_train)
# test_accuracy = best_model.score(X_test, y_test)
```

### 4. Custom PDF Class for Reporting

This example shows the custom `PDF` class that extends `fpdf2.FPDF` to add standardized headers and footers to all generated reports.

```python
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        """Custom header for the PDF report."""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Project Samantha - Comprehensive Analytics Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        """Custom footer with page numbering for the PDF report."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# Example Usage:
# pdf = PDF()
# pdf.add_page()
# pdf.set_font('Arial', '', 12)
# pdf.multi_cell(0, 10, "This is the content of the report.")
# pdf_output_bytes = pdf.output()
```

These examples provide a practical understanding of how the discussed data structures and algorithms are implemented within the `app.py` codebase.

## Testing Coverage Analysis

Upon reviewing the `app.py` codebase, it appears that there are **no explicit automated tests** (unit tests, integration tests, or end-to-end tests) implemented. The application is primarily designed for direct execution and interaction through the Streamlit framework, implying that testing is likely performed through manual UI interaction and observation.

### Current State of Testing:
-   **No Automated Test Suite:** There are no dedicated test files (e.g., `test_*.py`) or a testing framework (like `pytest`, `unittest`) configured or used within the project structure.
-   **Implicit Manual Testing:** The development and verification of features likely rely on:
    -   Running the `app.py` script and interacting with the Streamlit UI.
    -   Manually inputting data, triggering functions, and observing the output and application behavior.
    -   Visual inspection of charts, tables, and reports.
    -   Verifying database operations by checking the `samantha_data.db` file or through the application's data display features.

### Implications of Limited Automated Testing:
-   **Regression Risks:** Changes to the codebase (e.g., refactoring, bug fixes, new features) carry a higher risk of introducing regressions (breaking existing functionality) without immediate detection.
-   **Maintenance Challenges:** As the application grows, verifying all functionalities manually becomes increasingly time-consuming and error-prone.
-   **Code Quality Assurance:** Lack of tests can make it harder to ensure the correctness, robustness, and performance of individual functions and components, especially for complex algorithms like predictive modeling or data cleaning pipelines.
-   **Refactoring Difficulty:** Without tests to provide a safety net, refactoring code can be daunting, as it's difficult to confirm that changes haven't inadvertently altered behavior.
-   **Onboarding New Developers:** New contributors might find it challenging to understand the expected behavior of different parts of the system without executable specifications provided by tests.

### Areas Particularly Needing Testing:
-   **Core Algorithms:**
    -   `hash_password` and `verify_password`: Critical security functions that must be rigorously tested for correctness and edge cases.
    -   `check_password_strength`: Ensure all strength criteria are correctly applied.
    -   `authenticate_user`: Test various login scenarios (success, failure, locked account, expired password).
    -   `calculate_statistical_significance`, `calculate_effect_size`: Verify statistical correctness with known inputs.
    -   `perform_predictive_modeling`: Test data preparation, model training, prediction, and evaluation metrics.
    -   `get_comprehensive_outcome_data`: Crucial for data cleaning, imputation, and derived metrics. Test with various data quality issues and filter combinations.
-   **Database Interactions:**
    -   `init_database`: Ensure schema creation is correct.
    -   All `INSERT`, `UPDATE`, `DELETE`, and `SELECT` operations: Verify data integrity and correct retrieval.
-   **Report Generation:**
    -   `create_pdf_report` and `generate_*_report` functions: Ensure correct content, formatting, and chart embedding.
-   **Input Validation:**
    -   `validate_bulk_outcome_data`: Test with valid, invalid, and partially valid input data.



### 1. Password Hashing and Verification

### 2. Password Strength Check

- **Algorithm name:** Password Strength Check
- **Location in codebase:** `check_password_strength` function (lines 521-537)
- **Category:** Security, Input Validation
- **Purpose and Context:** Evaluates the strength of a given password against a set of predefined criteria (length, uppercase, lowercase, digit, special character). It's used during password reset or creation to enforce strong password policies.
- **Algorithm Description:**
    1. Initializes a dictionary `requirements` to track if each criterion is met.
    2. Checks password length (>= 8 characters).
    3. Uses regular expressions (`re.search`) to check for the presence of:
        - Uppercase letters (`[A-Z]`)
        - Lowercase letters (`[a-z]`)
        - Digits (`\d`)
        - Special characters (`[!@#$%^&*(),.?":{}|<>]`)
    4. Calculates a `score` based on the number of met requirements.
    5. Assigns a `strength` rating ('Very Weak' to 'Very Strong') based on the score.
    6. Determines `is_valid` based on a minimum score (>= 4).
- **Input/Output Specification:**
    - Input: `password` (str)
    - Output: `Dict` containing `score` (int), `strength` (str), `requirements` (Dict[str, bool]), and `is_valid` (bool).
- **Complexity Analysis:**
    - **Time complexity:** O(L) where L is the length of the password, due to regular expression searches. This is very fast for typical password lengths.
    - **Space complexity:** O(1) auxiliary space.
- **Implementation Details:** Uses Python's built-in string functions and the `re` module for pattern matching.
- **Dependencies:** `re`.
- **Correctness and Robustness:** Provides a clear, configurable set of rules for password strength. The use of regular expressions ensures accurate pattern matching.

### 3. User Authentication

- **Algorithm name:** User Authentication
- **Location in codebase:** `authenticate_user` function (lines 540-584)
- **Category:** Security, Access Control
- **Purpose and Context:** Verifies user credentials (username and password) against stored data, manages login attempts, account locking, and password expiration. It's the gateway for all authenticated access to the application.
- **Algorithm Description:**
    1. Queries the `users` table to retrieve user data based on the provided `username`, including `password_hash`, `login_attempts`, `account_locked`, `active` status, and `password_expires`.
    2. If no user is found, returns `None`.
    3. Checks if the account is `locked` or `inactive`; if so, returns `None`.
    4. Checks if the `password_expires` date has passed; if so, returns an error indicating password expiration.
    5. Calls `verify_password` to compare the provided password with the stored hash.
    6. If password verification is successful:
        - Resets `login_attempts` to 0.
        - Updates `last_login` timestamp.
        - Returns a dictionary with user details (`id`, `username`, `role`, `email`, `full_name`, `active`).
    7. If password verification fails:
        - Increments `login_attempts`.
        - Locks the account (`account_locked = TRUE`) if `login_attempts` reaches a predefined threshold (3 attempts).
        - Updates the `users` table with new `login_attempts` and `account_locked` status.
        - Returns `None`.
- **Input/Output Specification:**
    - Input: `username` (str), `password` (str)
    - Output: `Optional[Dict]` (user dictionary on success, `None` on failure, or `{'error': 'password_expired'}` if password expired).
- **Complexity Analysis:**
    - **Time complexity:** Dominated by the database query and the `verify_password` function. The database query is typically efficient due to indexing on `username`. `verify_password` is intentionally slow (see Algorithm 1).
    - **Space complexity:** O(1) auxiliary space.
- **Implementation Details:** Interacts with the DuckDB database, uses `verify_password` for cryptographic checks, and manages session state for authentication.
- **Dependencies:** `st.session_state.db_conn`, `duckdb`, `datetime`, `verify_password`.
- **Correctness and Robustness:** Implements standard security practices like account locking, password expiration, and secure password verification. Handles various failure scenarios gracefully.

### 4. Audit Logging

- **Algorithm name:** Audit Log Creation
- **Location in codebase:** `create_audit_log` function (lines 587-602)
- **Category:** Security, Data Integrity, Logging
- **Purpose and Context:** Records significant data changes (inserts, updates, deletes) in a dedicated `audit_log` table. This provides an immutable trail of who changed what, when, and why, which is crucial for compliance, debugging, and security.
- **Algorithm Description:**
    1. Generates a unique `log_id` using `uuid.uuid4()`.
    2. Retrieves the `user_id` of the currently authenticated user from `st.session_state`.
    3. Inserts a new record into the `audit_log` table with:
        - `log_id`
        - `table_name` (the table where the change occurred)
        - `record_id` (the ID of the record that was changed)
        - `action` (e.g., 'INSERT', 'UPDATE', 'DELETE')
        - `old_values` (serialized JSON of the record's state before the change, if applicable)
        - `new_values` (serialized JSON of the record's state after the change, if applicable)
        - `changed_by` (the user who made the change)
        - `change_reason` (an optional explanation for the change)
        - `timestamp` (defaults to `CURRENT_TIMESTAMP` in the database).
- **Input/Output Specification:**
    - Input: `table_name` (str), `record_id` (str), `action` (str), `old_values` (Dict, optional), `new_values` (Dict, optional), `reason` (str, optional)
    - Output: None (performs a database insert)
- **Complexity Analysis:**
    - **Time complexity:** O(1) for generating UUIDs and performing a single database insert. JSON serialization adds a small overhead proportional to the size of `old_values` or `new_values`.
    - **Space complexity:** O(1) auxiliary space for function execution. The `audit_log` table itself consumes space proportional to the number of log entries and the size of the `old_values`/`new_values` JSON strings.
- **Implementation Details:** Uses `uuid` for ID generation, `json.dumps` for serializing dictionary data, and DuckDB for persistence.
- **Dependencies:** `st.session_state.db_conn`, `duckdb`, `uuid`, `json`.
- **Correctness and Robustness:** Ensures that audit logs are created for specified actions, providing a robust mechanism for tracking data modifications. The use of `uuid` guarantees unique log entries.

### 5. Alert Generation

- **Algorithm name:** Alert Generation
- **Location in codebase:** `generate_alert` function (lines 605-618)
- **Category:** Notification, System Monitoring
- **Purpose and Context:** Creates system-wide alerts or notifications that can be targeted to specific users or roles. This is used to inform administrators or staff about important events, potential issues (e.g., data quality warnings), or required actions.
- **Algorithm Description:**
    1. Generates a unique `alert_id` using `uuid.uuid4()`.
    2. Sets an `expires_at` timestamp, typically 30 days from creation, to automatically clear old alerts.
    3. Inserts a new record into the `alerts` table with:
        - `alert_id`
        - `alert_type` (e.g., 'data_quality', 'security', 'system_status')
        - `severity` (e.g., 'info', 'warning', 'error')
        - `title` (a concise summary of the alert)
        - `message` (detailed description)
        - `target_user` (optional, specific user ID)
        - `target_role` (optional, specific role, e.g., 'Administrator')
        - `expires_at`
- **Input/Output Specification:**
    - Input: `alert_type` (str), `severity` (str), `title` (str), `message` (str), `target_user` (str, optional), `target_role` (str, optional)
    - Output: None (performs a database insert)
- **Complexity Analysis:**
    - **Time complexity:** O(1) for UUID generation and a single database insert.
    - **Space complexity:** O(1) auxiliary space.
- **Implementation Details:** Uses `uuid` for ID generation and DuckDB for persistence.
- **Dependencies:** `st.session_state.db_conn`, `duckdb`, `uuid`, `datetime`.
- **Correctness and Robustness:** Provides a flexible mechanism for system notifications. Alerts are automatically expired to prevent accumulation of stale information.

### 6. Statistical Significance Calculation

- **Algorithm name:** Statistical Significance Calculation (T-test, Mann-Whitney U test)
- **Location in codebase:** `calculate_statistical_significance` function (lines 621-633)
- **Category:** Statistical Analysis, Hypothesis Testing
- **Purpose and Context:** Determines if the difference between two datasets (e.g., outcome scores from two different interventions) is statistically significant. This helps in making data-driven decisions about intervention effectiveness.
- **Algorithm Description:**
    1. Takes two datasets (`data1`, `data2`) and a `test_type` (either 'ttest' or 'mannwhitney').
    2. If `test_type` is 'ttest', it performs an independent samples t-test using `scipy.stats.ttest_ind`.
    3. If `test_type` is 'mannwhitney', it performs a Mann-Whitney U test using `scipy.stats.mannwhitneyu` (non-parametric alternative).
    4. Returns the calculated statistic and p-value.
    5. Includes basic error handling to catch exceptions during calculation.
- **Input/Output Specification:**
    - Input: `data1` (array-like), `data2` (array-like), `test_type` (str, 'ttest' or 'mannwhitney')
    - Output: `Tuple[Optional[float], Optional[float]]` (statistic, p-value)
- **Complexity Analysis:**
    - **Time complexity:**
        - T-test: O(N1 + N2) where N1 and N2 are the sizes of the two datasets, as it involves calculating means and variances.
        - Mann-Whitney U test: O(N log N) where N is the total number of observations, primarily due to sorting.
    - **Space complexity:** O(1) auxiliary space (or O(N) for sorting in Mann-Whitney U test if a copy is made).
- **Implementation Details:** Leverages `scipy.stats` for robust statistical test implementations.
- **Dependencies:** `scipy.stats`.
- **Correctness and Robustness:** Provides standard statistical tests for comparing groups. The choice between t-test (parametric) and Mann-Whitney U (non-parametric) allows for flexibility based on data distribution assumptions.

### 7. Effect Size Calculation (Cohen's d)

- **Algorithm name:** Cohen's d Effect Size Calculation
- **Location in codebase:** `calculate_effect_size` function (lines 636-646)
- **Category:** Statistical Analysis, Effect Size Measurement
- **Purpose and Context:** Quantifies the magnitude of the difference between two groups, providing a standardized measure that complements statistical significance (p-value). A large effect size indicates a substantial difference, regardless of sample size.
- **Algorithm Description:**
    1. Takes two datasets (`data1`, `data2`).
    2. Calculates the pooled standard deviation, which is a weighted average of the standard deviations of the two groups. This is done using the formula:
       `pooled_std = sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))`
       where `n` is length and `var` is variance.
    3. Calculates Cohen's d using the formula:
       `effect_size = (mean1 - mean2) / pooled_std`
    4. Returns the calculated effect size.
    5. Includes basic error handling.
- **Input/Output Specification:**
    - Input: `data1` (array-like), `data2` (array-like)
    - Output: `Optional[float]` (Cohen's d effect size)
- **Complexity Analysis:**
    - **Time complexity:** O(N1 + N2) as it involves calculating means and variances of the two datasets.
    - **Space complexity:** O(1) auxiliary space.
- **Implementation Details:** Uses `numpy` for efficient numerical operations (mean, variance, sqrt).
- **Dependencies:** `numpy`.
- **Correctness and Robustness:** Implements the standard formula for Cohen's d. Provides a valuable metric for interpreting the practical significance of differences between groups.

### 8. Predictive Modeling (XGBoost Classifier Pipeline)

- **Algorithm name:** Predictive Modeling for Outcome Classification
- **Location in codebase:** `perform_predictive_modeling` function (lines 649-719)
- **Category:** Machine Learning, Classification, Predictive Analytics
- **Purpose and Context:** Builds and evaluates a machine learning model (XGBoost Classifier) to predict the category of normalized outcome scores ('Low', 'Medium', 'High') based on various intervention, demographic, and session-related features. This helps in proactive identification of individuals who might need additional support or interventions.
- **Algorithm Description:**
    1. **Feature Definition:** Identifies numeric (`cost_per_session`, `session_duration`, `quality_rating`) and categorical features (`intervention_name`, `intervention_category`, `metric_name`, `age_group`, `disability_category`, `support_level`).
    2. **Data Preparation:**
        - Selects relevant features and the target variable (`normalized_score`) from the input `outcome_data`.
        - Drops rows with missing `normalized_score`.
        - Creates a categorical target variable `y_labels` by binning `normalized_score` into 'Low', 'Medium', 'High' categories using `pd.cut`.
        - Encodes `y_labels` into numerical format using `LabelEncoder`.
        - One-hot encodes categorical features using `pd.get_dummies`.
        - Concatenates numeric and one-hot encoded categorical features into a single feature matrix `X`.
        - Handles potential remaining NaN values by logging an error.
    3. **Data Splitting:** Splits the data into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`) using `train_test_split` with a 70/30 ratio and stratification to maintain class balance.
    4. **Pipeline Creation:** Constructs a `Pipeline` consisting of:
        - `SelectKBest`: For feature selection, using `f_classif` as the scoring function.
        - `StandardScaler`: For scaling numeric features.
        - `XGBClassifier`: The core classification model.
    5. **Hyperparameter Tuning (GridSearchCV):**
        - Defines a `param_grid` with different values for `selector__k` (number of features), `model__n_estimators`, `model__max_depth`, and `model__learning_rate`.
        - Performs a `GridSearchCV` with 3-fold cross-validation to find the best combination of hyperparameters that maximizes accuracy.
    6. **Model Evaluation:**
        - Retrieves the `best_model` from `GridSearchCV`.
        - Calculates `train_accuracy` and `test_accuracy` using `best_model.score`.
    7. **Feature Importance:** Extracts the top 10 most important features from the `best_model` and stores them in a dictionary.
    8. **Return Results:** Returns a dictionary containing the trained `model`, `train_accuracy`, `test_accuracy`, `feature_importance`, `features_used` (column names of X), and the `label_encoder`.
    9. **Error Handling:** Catches and logs any exceptions during the modeling process.
- **Input/Output Specification:**
    - Input: `outcome_data` (pandas.DataFrame) - comprehensive outcome data.
    - Output: `Optional[Dict]` - A dictionary containing the trained model, accuracies, feature importances, features used, and label encoder on success; `None` on failure or insufficient data.
- **Complexity Analysis:**
    - **Time complexity:**
        - Data preparation: Dominated by one-hot encoding, which is O(N * M) where N is number of rows and M is number of categorical features.
        - `train_test_split`: O(N).
        - `GridSearchCV`: O(P * C * N * T * F) where P is number of parameter combinations, C is cross-validation folds, N is number of samples, T is number of trees, and F is number of features. XGBoost training itself is typically O(N * T * F). This can be computationally intensive.
    - **Space complexity:** O(N * F) for storing the feature matrix `X` and intermediate data structures during model training.
- **Implementation Details:** Utilizes `pandas` for data manipulation, `sklearn` for pipeline, scaling, feature selection, and model selection, and `xgboost` for the classification model.
- **Dependencies:** `pandas`, `numpy`, `scipy.stats`, `sklearn.ensemble`, `sklearn.model_selection`, `sklearn.preprocessing`, `sklearn.pipeline`, `xgboost`.
- **Correctness and Robustness:**
    - Employs a robust machine learning pipeline including feature selection, scaling, and hyperparameter tuning to prevent overfitting and improve generalization.
    - Uses `XGBClassifier`, known for its performance and ability to handle various data types.
    - Stratified splitting ensures representative training and test sets.
    - Includes checks for sufficient data and NaN values.
- **Performance Characteristics:** Designed for accuracy and robustness, which can be computationally intensive, especially `GridSearchCV`. The `n_jobs=-1` parameter in `GridSearchCV` attempts to parallelize the process across all available CPU cores.

### 9. Comprehensive Outcome Data Cleaning and Imputation

- **Algorithm name:** Data Cleaning and Imputation Pipeline
- **Location in codebase:** `get_comprehensive_outcome_data` function (lines 780-829)
- **Category:** Data Preprocessing, Data Quality, ETL
- **Purpose and Context:** This pipeline is responsible for retrieving, cleaning, imputing missing values, and calculating derived metrics from raw outcome and cost data. It ensures that the data used for analytics and machine learning is consistent, complete, and correctly formatted.
- **Algorithm Description:**
    1. **Data Retrieval:** Executes a SQL query to join `outcome_records`, `interventions`, `outcome_metrics`, `individuals`, `users`, and `cost_records` tables. Applies optional date range and categorical filters.
    2. **Step 1: Type Conversion:** Iterates through a predefined list of `numeric_cols` and attempts to convert them to numeric types using `pd.to_numeric`. Errors are coerced to `NaN`.
    3. **Step 2: Imputation:**
        - For numeric columns (`numeric_cols_to_impute`), `NaN` values are filled with the column's median.
        - For categorical columns (`categorical_cols_to_impute`), `NaN` values are filled with the column's mode. If the mode is empty (e.g., all NaNs), it defaults to "Unknown".
    4. **Step 3: Derived Metrics Calculation:**
        - Calculates `total_cost` by summing `direct_cost`, `indirect_cost`, `overhead_cost`, and `material_cost`.
        - Calculates `cost_variance` and `duration_variance`.
        - Calculates `normalized_score` (0-10 scale) based on `score`, `scale_min`, `scale_max`, and `higher_is_better` flags. It clamps the score between 0 and 10 and fills any remaining NaNs with the median.
        - Calculates `improvement_from_baseline` based on `score` and `baseline_score`, considering `higher_is_better`.
- **Input/Output Specification:**
    - Input: `date_range` (Tuple[date, date], optional), `filters` (Dict, optional)
    - Output: `pandas.DataFrame` - a cleaned, imputed, and enriched DataFrame of outcome data.
- **Complexity Analysis:**
    - **Time complexity:**
        - SQL Query: Depends on database size and query complexity, typically optimized by DuckDB.
        - Type Conversion: O(N * C_numeric) where N is rows, C_numeric is number of numeric columns.
        - Imputation: O(N * C_impute) for median/mode calculations.
        - Derived Metrics: O(N * C_derived) for vectorized column operations.
        - Overall: Dominated by the SQL query and vectorized pandas operations, generally efficient.
    - **Space complexity:** O(N * F) where N is rows and F is number of columns, for storing the DataFrame and intermediate copies.
- **Implementation Details:** Heavily relies on `pandas` for efficient DataFrame operations and `duckdb` for data retrieval.
- **Dependencies:** `pandas`, `duckdb`, `numpy`.
- **Correctness and Robustness:**
    - Provides a systematic approach to handle missing and inconsistent data.
    - Normalizes outcome scores to a common scale, enabling fair comparisons across different metrics.
    - Imputation strategies (median for numeric, mode for categorical) are chosen to minimize distortion.
    - Includes error handling for database queries.
- **Performance Characteristics:** Designed for efficient processing of tabular data. Vectorized operations in pandas are highly optimized in C, making this pipeline performant for moderately large datasets.
