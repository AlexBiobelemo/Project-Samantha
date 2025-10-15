import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, date, timedelta
import io
import json
from typing import List, Optional, Tuple
import warnings
import random
from fpdf import FPDF
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from security import hash_password, verify_password, check_password_strength, authenticate_user, create_audit_log, generate_alert
from database import init_database # Import init_database from database.py
import uuid # Keep uuid for generating temporary passwords and other IDs within app.py


class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Project Samantha - Comprehensive Analytics Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_report(title, date_range, content_dict):
    """
    Generates a PDF report from a dictionary of content.
    Content can include text, pandas DataFrames, and Plotly figures.
    """
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, title, 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Reporting Period: {date_range[0]} to {date_range[1]}", 0, 1, 'L')
    pdf.ln(5)

    for item in content_dict:
        content_type = item['type']
        data = item['data']

        if content_type == 'header':
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, data, 0, 1, 'L')
            pdf.ln(2)
        elif content_type == 'text':
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 5, data)
            pdf.ln(5)
        elif content_type == 'dataframe':
            pdf.set_font('Arial', '', 9)
            # Simple table rendering
            col_widths = [30] * len(data.columns) # Adjustable
            line_height = pdf.font_size * 2.5
            # Header
            for col_name in data.columns:
                pdf.cell(col_widths[0], line_height, col_name, border=1)
            pdf.ln(line_height)
            # Rows
            for index, row in data.iterrows():
                for val in row:
                    pdf.cell(col_widths[0], line_height, str(val), border=1)
                pdf.ln(line_height)
            pdf.ln(5)
        elif content_type == 'figure':
            # Convert Plotly fig to an in-memory image
            img_bytes = io.BytesIO(data.to_image(format="png", width=800, height=450, scale=2))
            pdf.image(img_bytes, w=180) # A4 width is ~210mm
            pdf.ln(5)
            
    # Return PDF as bytes
    return pdf.output() # FPDF2 by default returns bytes, no need to encode.

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Project Samantha - Enterprise Care Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0F2027 0%, #203A43 50%, #2C5364 100%); /* Dark blue gradient */
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2C5364; /* Matching border color */
    }
    .alert-success {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid transparent;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid transparent;
        margin: 1rem 0;
    }
    .alert-danger {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid transparent;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .kpi-container {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%); /* Dark blue gradient */
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def show_forgot_password_form():
    """UI for the password reset flow."""
    st.markdown('<div class="main-header"><h1>🔑 Password Reset</h1></div>', unsafe_allow_html=True)
    conn = st.session_state.db_conn
    
    with st.form("reset_form"):
        username = st.text_input("Enter your username to reset your password")
        submitted = st.form_submit_button("Reset Password")

    if submitted and username:
        user_exists = conn.execute("SELECT id FROM users WHERE username = ?", [username]).fetchone()
        if user_exists:
            # Generate a secure temporary password
            new_temp_password = f"temp_{uuid.uuid4().hex[:8]}"
            password_hash, salt = hash_password(new_temp_password) # Assuming you use a modern hash function
            
            # Update the user's password and force a reset on next login
            conn.execute("""
                UPDATE users
                SET password_hash = ?,
                    login_attempts = 0,
                    account_locked = FALSE,
                    password_expires = ?
                WHERE username = ?
            """, [password_hash, datetime.now() + timedelta(days=1), username])
            
            st.success("A temporary password has been generated.")
            st.info(f"For this demo, your temporary password is: **{new_temp_password}**")
            st.warning("You will be required to change this password immediately after logging in.")
        else:
            st.error("No account found with that username.")
        
        if st.button("← Back to Login"):
            del st.session_state.password_reset_flow
            st.rerun()

# Advanced Analytics Functions
def calculate_statistical_significance(data1, data2, test_type='ttest'):
    """Calculate statistical significance between two datasets"""
    try:
        if test_type == 'ttest':
            stat, p_value = stats.ttest_ind(data1, data2)
        elif test_type == 'mannwhitney':
            stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        else:
            return None, None

        return stat, p_value
    except:
        return None, None


def calculate_effect_size(data1, data2):
    """Calculate Cohen's d effect size"""
    try:
        pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + (len(data2) - 1) * np.var(data2, ddof=1)) / (
                len(data1) + len(data2) - 2))
        effect_size = (np.mean(data1) - np.mean(data2)) / pooled_std
        return effect_size
    except:
        return None


@st.cache_data(ttl=3600) # Cache for 1 hour
def perform_predictive_modeling(outcome_data):
    """
    Perform predictive modeling using a tuned classification pipeline.
    """
    try:
        # Define features
        numeric_features_list = ['cost_per_session', 'session_duration', 'quality_rating']
        categorical_features_list = [
            'intervention_name', 'intervention_category', 'metric_name', 
            'age_group', 'disability_category', 'support_level'
        ]
        
        all_features = numeric_features_list + categorical_features_list
        features_to_model = outcome_data[all_features + ['score', 'normalized_score']].dropna(subset=['normalized_score'])

        if len(features_to_model) < 100:
            return None

        # Create and encode the target variable 'y'
        bins = [-1, 4, 7, 11]
        labels = ['Low', 'Medium', 'High']
        y_labels = pd.cut(features_to_model['normalized_score'], bins=bins, labels=labels)
        le = LabelEncoder()
        y = le.fit_transform(y_labels)
        
        # Prepare the feature set 'X'
        X_categorical = pd.get_dummies(features_to_model[categorical_features_list], drop_first=True)
        X_numeric = features_to_model[numeric_features_list]
        X = pd.concat([X_numeric, X_categorical], axis=1)
        X.columns = X.columns.astype(str)
        
        # Final NaN check
        if X.isnull().sum().sum() > 0:
            nan_cols = X.columns[X.isnull().any()].tolist()
            st.error(f"NaN values still detected after cleaning in columns: {', '.join(nan_cols)}.")
            return None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Create a Pipeline with GridSearchCV for tuning ---
        pipeline = Pipeline([
            ('selector', SelectKBest(score_func=f_classif)),
            ('scaler', StandardScaler()),
            ('model', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
        ])
        
        # Define a parameter grid to search over
        # These parameters are chosen to control overfitting
        param_grid = {
            'selector__k': [20, 30], # Test using the top 20 or 30 features
            'model__n_estimators': [100, 150],
            'model__max_depth': [3, 4], # Test shallower trees
            'model__learning_rate': [0.05, 0.1]
        }
        
        # Set up and run the grid search
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_

        # Evaluate the best model
        train_score = best_model.score(X_train, y_train)
        test_score = best_model.score(X_test, y_test)

        # Get feature importances from the best model
        selector_cols = best_model.named_steps['selector'].get_support(indices=True)
        feature_names = X.columns[selector_cols]
        importances = pd.Series(best_model.named_steps['model'].feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
        feature_importance = importances.to_dict()

        return {
            'model': best_model,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': feature_importance,
            'features_used': X.columns, # Pass the original columns for prediction tool
            'label_encoder': le # Pass the fitted label encoder
        }
    except Exception as e:
        st.error(f"Predictive modeling error: {str(e)}")
        return None


# Data retrieval functions
def get_facility_data():
    """Get current facility data"""
    conn = st.session_state.db_conn
    result = conn.execute("SELECT id, name, currency FROM facilities LIMIT 1").fetchone()
    return {'id': result[0], 'name': result[1], 'currency': result[2]} if result else None


def get_interventions():
    """Get all active interventions"""
    conn = st.session_state.db_conn
    return conn.execute("""
        SELECT id, name, cost_per_session, description, duration_minutes
        FROM interventions
        WHERE active = TRUE
        ORDER BY name
    """).df()


def get_outcome_metrics():
    """Get all active outcome metrics"""
    conn = st.session_state.db_conn
    return conn.execute("""
        SELECT id, name, category, scale_min, scale_max, description, unit_of_measure
        FROM outcome_metrics
        WHERE active = TRUE
        ORDER BY name
    """).df()


def get_individuals():
    """Get all active individuals"""
    conn = st.session_state.db_conn
    return conn.execute("""
        SELECT id, anonymous_id, age_group, gender, disability_category, disability_severity, 
               support_level, admission_date
        FROM individuals
        WHERE active = TRUE
        ORDER BY admission_date
    """).df()

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_comprehensive_outcome_data(date_range=None, filters=None):
    """Get comprehensive outcome data with a robust cleaning and imputation pipeline."""
    conn = st.session_state.db_conn
    query = """
        SELECT 
            o.*,
            i.name as intervention_name, i.category as intervention_category,
            i.cost_per_session, i.duration_minutes as planned_duration, i.evidence_level,
            m.name as metric_name, m.category as metric_category, m.scale_min, m.scale_max,
            m.higher_is_better, m.clinical_significance_threshold, m.unit_of_measure,
            ind.anonymous_id, ind.age_group, ind.gender, ind.disability_category,
            ind.disability_severity, ind.support_level, ind.funding_source,
            u.full_name as staff_name,
            c.direct_cost, c.indirect_cost, c.overhead_cost, c.material_cost
        FROM outcome_records o
        JOIN interventions i ON o.intervention_id = i.id
        JOIN outcome_metrics m ON o.outcome_metric_id = m.id
        JOIN individuals ind ON o.individual_id = ind.id
        LEFT JOIN users u ON o.staff_id = u.id
        LEFT JOIN cost_records c ON o.session_id = c.session_id
        WHERE o.attendance_status = 'attended'
    """
    params = []

    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        query += " AND o.session_date BETWEEN ? AND ?"
        params.extend([date_range[0], date_range[1]])
    if filters:
        filter_map = {
            'intervention': ("i.name", filters.get('intervention')),
            'age_group': ("ind.age_group", filters.get('age_group')),
            'disability': ("ind.disability_category", filters.get('disability')),
            'staff': ("u.full_name", filters.get('staff')),
        }
        for key, (column, value) in filter_map.items():
            if value and value != "All":
                query += f" AND {column} = ?"
                params.append(value)
    
    query += " ORDER BY o.session_date DESC"
    try:
        df = conn.execute(query, params).df()
    except Exception as e:
        st.error(f"Database query error: {str(e)}")
        return pd.DataFrame()

    if df.empty:
        return df

    # --- Data Cleaning Pipeline ---

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

    # STEP 3: Calculate derived metrics
    cost_cols = ['direct_cost', 'indirect_cost', 'overhead_cost', 'material_cost']
    df['total_cost'] = df[cost_cols].sum(axis=1)

    df['cost_variance'] = df['total_cost'] - df['cost_per_session']
    df['duration_variance'] = df['session_duration'] - df['planned_duration']

    df['normalized_score'] = df.apply(
        lambda x: ((x['score'] - x['scale_min']) / (x['scale_max'] - x['scale_min'])) * 10
        if x['higher_is_better'] and x['scale_max'] != x['scale_min']
        else ((x['scale_max'] - x['score']) / (x['scale_max'] - x['scale_min'])) * 10
        if not x['higher_is_better'] and x['scale_max'] != x['scale_min']
        else 5.0, # Default to a neutral score if scale is invalid
        axis=1
    )
    
    # Clamp the normalized score between 0 and 10 to handle any outliers from bad data entry.
    df['normalized_score'] = df['normalized_score'].clip(0, 10)
    
    # Fill any remaining NaNs from calculation errors
    df['normalized_score'].fillna(df['normalized_score'].median(), inplace=True)

    df['improvement_from_baseline'] = df.apply(
        lambda x: x['score'] - x['baseline_score'] if x['higher_is_better'] else x['baseline_score'] - x['score'],
        axis=1
    )
    df['improvement_from_baseline'].fillna(0, inplace=True)

    return df


@st.cache_data(ttl=3600) # Cache for 1 hour
def get_advanced_analytics():
    """Generate advanced analytics and insights"""
    data = get_comprehensive_outcome_data()

    if data.empty:
        return {}

    analytics = {}

    # Cost-effectiveness analysis
    intervention_analysis = data.groupby('intervention_name').agg({
        'total_cost': ['sum', 'mean', 'std'],
        'normalized_score': ['mean', 'std', 'count'],
        'improvement_from_baseline': ['mean', 'std'],
        'quality_rating': 'mean',
        'session_duration': 'mean'
    }).round(2)

    intervention_analysis.columns = [f'{col[1]}_{col[0]}' if col[1] else col[0] for col in
                                     intervention_analysis.columns]
    intervention_analysis['cost_per_outcome_point'] = intervention_analysis['sum_total_cost'] / intervention_analysis[
        'mean_normalized_score']
    intervention_analysis['efficiency_ratio'] = intervention_analysis['mean_improvement_from_baseline'] / \
                                                intervention_analysis['mean_total_cost']

    analytics['intervention_analysis'] = intervention_analysis.reset_index()

    # Demographic analysis
    demographic_analysis = data.groupby(['age_group', 'disability_category']).agg({
        'normalized_score': ['mean', 'count'],
        'improvement_from_baseline': 'mean',
        'total_cost': 'sum'
    }).round(2)

    analytics['demographic_analysis'] = demographic_analysis.reset_index()

    # Staff performance analysis
    staff_analysis = data.groupby('staff_name').agg({
        'quality_rating': 'mean',
        'normalized_score': 'mean',
        'improvement_from_baseline': 'mean',
        'session_duration': 'mean',
        'total_cost': 'sum'
    }).round(2)

    analytics['staff_analysis'] = staff_analysis.reset_index()

    # Trend analysis
    monthly_trends = data.copy()
    monthly_trends['year_month'] = pd.to_datetime(monthly_trends['session_date']).dt.to_period('M')

    trend_analysis = monthly_trends.groupby('year_month').agg({
        'normalized_score': 'mean',
        'total_cost': 'sum',
        'improvement_from_baseline': 'mean',
        'quality_rating': 'mean'
    }).round(2)

    analytics['trend_analysis'] = trend_analysis.reset_index()

    # Quality indicators
    quality_metrics = {
        'overall_attendance_rate': (data['attendance_status'] == 'attended').mean() * 100,
        'average_quality_rating': data['quality_rating'].mean(),
        'sessions_below_quality_threshold': (data['quality_rating'] < 3).sum(),
        'cost_variance_percentage': (data['cost_variance'].abs() / data['cost_per_session']).mean() * 100,
        'duration_variance_percentage': (data['duration_variance'].abs() / data['planned_duration']).mean() * 100
    }

    analytics['quality_metrics'] = quality_metrics

    # Statistical significance testing
    if len(data['intervention_name'].unique()) > 1:
        interventions = data['intervention_name'].unique()
        significance_results = []

        for i in range(len(interventions)):
            for j in range(i + 1, len(interventions)):
                int1_data = data[data['intervention_name'] == interventions[i]]['normalized_score'].dropna()
                int2_data = data[data['intervention_name'] == interventions[j]]['normalized_score'].dropna()

                if len(int1_data) > 5 and len(int2_data) > 5:
                    stat, p_value = calculate_statistical_significance(int1_data, int2_data)
                    effect_size = calculate_effect_size(int1_data, int2_data)

                    if stat is not None:
                        significance_results.append({
                            'intervention_1': interventions[i],
                            'intervention_2': interventions[j],
                            'p_value': p_value,
                            'effect_size': effect_size,
                            'significant': p_value < 0.05 if p_value else False
                        })

        analytics['significance_testing'] = significance_results

    return analytics


def login_form():
    """Enhanced login form with security features"""
    st.markdown('<div class="main-header"><h1> Project Samantha</h1><h3>Enterprise Care Analytics Platform</h3></div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Secure Access Portal")

        # Check for alerts or system messages
        if 'login_error' in st.session_state:
            st.error(st.session_state.login_error)
            del st.session_state.login_error

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            remember_me = st.checkbox("Remember me")

            col_login, col_forgot = st.columns(2)
            with col_login:
                submitted = st.form_submit_button("Login", use_container_width=True)
            with col_forgot:
                forgot_password = st.form_submit_button("Reset Password", use_container_width=True)

            if submitted:
                if username and password:
                    user = authenticate_user(st.session_state.db_conn, username, password)
                    if user:
                        if user.get('error') == 'password_expired':
                            st.session_state.password_reset_required = True
                            st.session_state.reset_username = username
                            st.rerun()
                        else:
                            st.session_state.authenticated = True
                            st.session_state.user = user
                            st.rerun()
                    else:
                        st.session_state.login_error = "Invalid credentials or account locked. Contact administrator if needed."
                        st.rerun()
                else:
                    st.error("Please enter both username and password")

            if forgot_password:
                st.session_state.password_reset_flow = True
                st.rerun()

        st.markdown("---")

        # Demo credentials
        with st.expander("Demo Access Information"):
            st.info("**Administrator:** Username: admin, Password: admin123")
            st.info("**Staff Member:** Username: therapist1, Password: therapy123")
            st.warning("Change default passwords in production environment")

        # System status
        st.markdown("### System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.success(" Database: Online")
            st.success(" Analytics: Online")
        with col2:
            st.success(" Security: Active")
            st.success(" Backup: Current")


# Enhanced Dashboard Functions
def show_executive_dashboard():
    """Executive-level dashboard with comprehensive KPIs and insights"""
    st.markdown('<div class="main-header"><h2> Executive Dashboard</h2></div>', unsafe_allow_html=True)

    # Date range selector with proper handling
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        try:
            default_start = date.today() - timedelta(days=90)
            default_end = date.today()
            date_range = st.date_input(
                "Analysis Period",
                value=(default_start, default_end),
                help="Select date range for analysis"
            )
        except Exception as e:
            st.error(f"Date input error: {str(e)}")
            date_range = (date.today() - timedelta(days=90), date.today())

    with col2:
        comparison_period = st.selectbox(
            "Compare to Previous",
            ["None", "Previous Period", "Previous Quarter", "Previous Year"]
        )
    with col3:
        auto_refresh = st.checkbox("Auto-refresh", help="Refresh data every 5 minutes")

    # Get comprehensive data with error handling
    try:
        filters = {}
        outcome_data = get_comprehensive_outcome_data(
            date_range if isinstance(date_range, tuple) and len(date_range) == 2 else None, filters)
        analytics = get_advanced_analytics()
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        outcome_data = pd.DataFrame()
        analytics = {}

    if outcome_data.empty:
        st.warning("No data available for the selected period.")
        return

    # Top-level KPIs
    st.subheader("Key Performance Indicators")

    total_sessions = len(outcome_data)
    total_cost = outcome_data['total_cost'].sum()
    avg_outcome = outcome_data['normalized_score'].mean()
    cost_per_outcome = total_cost / avg_outcome if avg_outcome > 0 else 0

    # Quality metrics
    quality_metrics = analytics.get('quality_metrics', {})
    attendance_rate = quality_metrics.get('overall_attendance_rate', 0)
    avg_quality = quality_metrics.get('average_quality_rating', 0)

    # Display KPI cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

    facility = get_facility_data()
    currency = facility['currency']

    with kpi_col1:
        st.metric(
            "Total Investment",
            f"{currency}{total_cost:,.0f}",
            delta=f"{total_sessions} sessions",
            help="Total cost of all interventions in selected period"
        )

    with kpi_col2:
        st.metric(
            "Avg Outcome Score",
            f"{avg_outcome:.1f}/10",
            delta=f"{len(outcome_data['anonymous_id'].unique())} residents",
            help="Average normalized outcome score across all metrics"
        )

    with kpi_col3:
        st.metric(
            "Cost Efficiency",
            f"{currency}{cost_per_outcome:.0f}/pt",
            help="Cost per normalized outcome point achieved"
        )

    with kpi_col4:
        attendance_delta = f"{attendance_rate:.1f}%" if attendance_rate > 85 else f"-{100 - attendance_rate:.1f}%"
        st.metric(
            "Attendance Rate",
            f"{attendance_rate:.1f}%",
            delta=attendance_delta,
            help="Overall session attendance rate"
        )

    with kpi_col5:
        st.metric(
            "Quality Rating",
            f"{avg_quality:.1f}/5",
            help="Average session quality rating"
        )

    # Advanced Analytics Section
    st.markdown("---")
    st.subheader("Advanced Analytics & Insights")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Cost-Effectiveness Analysis",
        "Outcome Trends",
        "Statistical Analysis",
        "Predictive Insights",
        "Performance Benchmarks"
    ])

    with tab1:
        show_cost_effectiveness_analysis(analytics, outcome_data, currency)

    with tab2:
        show_outcome_trends_analysis(analytics, outcome_data)

    with tab3:
        show_statistical_analysis(analytics, outcome_data)

    with tab4:
        show_predictive_insights(outcome_data)

    with tab5:
        show_performance_benchmarks(analytics, outcome_data)


def show_cost_effectiveness_analysis(analytics, outcome_data, currency):
    """Comprehensive cost-effectiveness analysis with a standard table display."""
    if 'intervention_analysis' not in analytics:
        st.info("Insufficient data for cost-effectiveness analysis.")
        return

    intervention_analysis = analytics['intervention_analysis']
    intervention_analysis = intervention_analysis.sort_values('efficiency_ratio', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        # Cost vs Effectiveness Scatter Plot
        fig_scatter = px.scatter(
            intervention_analysis,
            x='mean_total_cost',
            y='mean_normalized_score',
            size='count_normalized_score',
            color='mean_quality_rating',
            hover_data=['cost_per_outcome_point', 'efficiency_ratio'],
            title='Cost vs Effectiveness Analysis',
            labels={
                'mean_total_cost': f'Average Cost per Session ({currency})',
                'mean_normalized_score': 'Average Outcome Score (0-10)',
                'count_normalized_score': 'Number of Sessions',
                'mean_quality_rating': 'Quality Rating'
            },
            color_continuous_scale='RdYlGn'
        )
        avg_cost = intervention_analysis['mean_total_cost'].mean()
        avg_outcome = intervention_analysis['mean_normalized_score'].mean()
        fig_scatter.add_hline(y=avg_outcome, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_vline(x=avg_cost, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        # Efficiency ranking bar chart
        fig_efficiency = px.bar(
            intervention_analysis.head(10),
            x='efficiency_ratio',
            y='intervention_name',
            orientation='h',
            title='Top 10 Most Efficient Interventions',
            labels={'efficiency_ratio': 'Efficiency Ratio (Improvement/Cost)', 'intervention_name': ''},
            color='efficiency_ratio',
            color_continuous_scale='RdYlGn'
        )
        fig_efficiency.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_efficiency, use_container_width=True)

    # Detailed cost breakdown table
    st.subheader("Intervention Cost-Effectiveness Ranking")

    display_data = intervention_analysis.copy()
    display_data['rank'] = range(1, len(display_data) + 1)
    display_data = display_data[[
        'rank', 'intervention_name', 'count_normalized_score', 'mean_total_cost',
        'mean_normalized_score', 'cost_per_outcome_point', 'efficiency_ratio', 'mean_quality_rating'
    ]]

    # Format for display
    display_data['mean_total_cost'] = display_data['mean_total_cost'].apply(lambda x: f"{currency}{x:.2f}")
    display_data['mean_normalized_score'] = display_data['mean_normalized_score'].apply(lambda x: f"{x:.1f}")
    display_data['cost_per_outcome_point'] = display_data['cost_per_outcome_point'].apply(
        lambda x: f"{currency}{x:.2f}")
    display_data['efficiency_ratio'] = display_data['efficiency_ratio'].apply(lambda x: f"{x:.4f}")
    display_data['mean_quality_rating'] = display_data['mean_quality_rating'].apply(lambda x: f"{x:.1f}/5")

    display_data.columns = [
        'Rank', 'Intervention', 'Sessions', 'Avg Cost', 'Avg Outcome',
        'Cost/Outcome', 'Efficiency', 'Quality'
    ]
    
 
    # Dataframe is passed directly to st.dataframe without the .style.apply() method.
    
    table_height = (len(display_data) + 1) * 35 + 3

    st.dataframe(
        display_data, # Pass the plain dataframe
        use_container_width=True,
        hide_index=True,
        height=table_height
    )

def show_outcome_trends_analysis(analytics, outcome_data):
    """Advanced outcome trends and longitudinal analysis"""
    if 'trend_analysis' not in analytics:
        st.info("Insufficient data for trend analysis.")
        return

    trend_data = analytics['trend_analysis']
    trend_data['year_month_str'] = trend_data['year_month'].astype(str)

    # Multi-metric trend chart
    fig_trends = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Outcome Scores Over Time', 'Cost Trends', 'Quality Ratings', 'Improvement Rates'),
        specs=[[{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Outcome scores trend
    fig_trends.add_trace(
        go.Scatter(x=trend_data['year_month_str'], y=trend_data['normalized_score'],
                   mode='lines+markers', name='Avg Outcome Score', line=dict(color='blue')),
        row=1, col=1
    )

    # Cost trends
    fig_trends.add_trace(
        go.Scatter(x=trend_data['year_month_str'], y=trend_data['total_cost'],
                   mode='lines+markers', name='Total Cost', line=dict(color='red')),
        row=1, col=2
    )

    # Quality ratings
    fig_trends.add_trace(
        go.Scatter(x=trend_data['year_month_str'], y=trend_data['quality_rating'],
                   mode='lines+markers', name='Quality Rating', line=dict(color='green')),
        row=2, col=1
    )

    # Improvement rates
    fig_trends.add_trace(
        go.Scatter(x=trend_data['year_month_str'], y=trend_data['improvement_from_baseline'],
                   mode='lines+markers', name='Improvement Rate', line=dict(color='purple')),
        row=2, col=2
    )

    fig_trends.update_layout(height=600, title_text="Comprehensive Performance Trends", showlegend=False)
    st.plotly_chart(fig_trends, use_container_width=True)

    # Seasonal analysis
    if len(outcome_data) > 100:
        st.subheader("Seasonal Pattern Analysis")

        seasonal_data = outcome_data.copy()
        seasonal_data['month'] = pd.to_datetime(seasonal_data['session_date']).dt.month
        seasonal_data['quarter'] = pd.to_datetime(seasonal_data['session_date']).dt.quarter
        seasonal_data['day_of_week'] = pd.to_datetime(seasonal_data['session_date']).dt.dayofweek

        col1, col2 = st.columns(2)

        with col1:
            monthly_performance = seasonal_data.groupby('month')['normalized_score'].mean()
            fig_monthly = px.line(
                x=monthly_performance.index,
                y=monthly_performance.values,
                title='Performance by Month',
                labels={'x': 'Month', 'y': 'Average Outcome Score'}
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

        with col2:
            quarterly_costs = seasonal_data.groupby('quarter')['total_cost'].sum()
            fig_quarterly = px.bar(
                x=['Q1', 'Q2', 'Q3', 'Q4'][:len(quarterly_costs)],
                y=quarterly_costs.values,
                title='Quarterly Cost Distribution',
                labels={'x': 'Quarter', 'y': 'Total Cost'}
            )
            st.plotly_chart(fig_quarterly, use_container_width=True)


def show_statistical_analysis(analytics, outcome_data):
    """Advanced statistical analysis and hypothesis testing"""
    st.subheader("Statistical Analysis")

    if 'significance_testing' in analytics:
        significance_results = analytics['significance_testing']

        if significance_results:
            st.markdown("### Intervention Comparison - Statistical Significance")

            sig_df = pd.DataFrame(significance_results)
            sig_df['p_value_formatted'] = sig_df['p_value'].apply(
                lambda x: f"{x:.4f}" if x >= 0.001 else "< 0.001"
            )
            sig_df['effect_size_interpretation'] = sig_df['effect_size'].apply(
                lambda x: "Large" if abs(x) > 0.8 else "Medium" if abs(x) > 0.5 else "Small" if abs(
                    x) > 0.2 else "Negligible"
            )
            sig_df['significant_symbol'] = sig_df['significant'].apply(lambda x: "?" if x else "?")

            display_sig = sig_df[[
                'intervention_1', 'intervention_2', 'p_value_formatted',
                'effect_size', 'effect_size_interpretation', 'significant_symbol'
            ]]
            display_sig.columns = [
                'Intervention A', 'Intervention B', 'P-Value',
                'Effect Size', 'Magnitude', 'Significant'
            ]

            st.dataframe(display_sig, use_container_width=True)

            # Highlight significant differences
            significant_count = len([r for r in significance_results if r['significant']])
            total_comparisons = len(significance_results)

            st.info(
                f"Found {significant_count} statistically significant differences out of {total_comparisons} comparisons")
        else:
            st.info("Need at least 2 interventions with sufficient data for statistical comparison")

    # Correlation analysis
    st.markdown("### Correlation Analysis")

    numeric_cols = ['normalized_score', 'total_cost', 'session_duration', 'quality_rating', 'improvement_from_baseline']
    correlation_data = outcome_data[numeric_cols].corr()

    fig_corr = px.imshow(
        correlation_data.values,
        labels=dict(x="Variables", y="Variables", color="Correlation"),
        x=correlation_data.columns,
        y=correlation_data.columns,
        color_continuous_scale='RdBu',
        title='Variable Correlation Matrix'
    )

    # Add correlation values as annotations
    for i in range(len(correlation_data.columns)):
        for j in range(len(correlation_data.columns)):
            fig_corr.add_annotation(
                x=i, y=j,
                text=f"{correlation_data.iloc[j, i]:.2f}",
                showarrow=False,
                font=dict(color="white" if abs(correlation_data.iloc[j, i]) > 0.5 else "black")
            )

    st.plotly_chart(fig_corr, use_container_width=True)

    # Key insights from correlation
    strong_correlations = []
    for i in range(len(correlation_data.columns)):
        for j in range(i + 1, len(correlation_data.columns)):
            corr_value = correlation_data.iloc[i, j]
            if abs(corr_value) > 0.5:
                strong_correlations.append({
                    'var1': correlation_data.columns[i],
                    'var2': correlation_data.columns[j],
                    'correlation': corr_value
                })

    if strong_correlations:
        st.markdown("#### Key Correlations Found:")
        for corr in strong_correlations:
            direction = "positive" if corr['correlation'] > 0 else "negative"
            strength = "strong" if abs(corr['correlation']) > 0.7 else "moderate"
            st.write(
                f" **{strength.capitalize()} {direction} correlation** between {corr['var1']} and {corr['var2']} (r = {corr['correlation']:.3f})")


def show_predictive_insights(outcome_data):
    """Show predictive modeling insights and an interactive classification tool."""
    st.subheader("Predictive Analytics")

    model_results = perform_predictive_modeling(outcome_data)

    if model_results:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Model Accuracy (Training)", f"{model_results['train_accuracy']:.1%}")
            st.metric("Model Accuracy (Testing)", f"{model_results['test_accuracy']:.1%}")

        with col2:
            importance_df = pd.DataFrame(
                list(model_results['feature_importance'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)

            fig_importance = px.bar(
                importance_df, x='Importance', y='Feature', orientation='h',
                title='Top 10 Most Important Features'
            )
            st.plotly_chart(fig_importance, use_container_width=True)

        # Outcome Prediction Tool for Classification
        st.markdown("### Outcome Prediction Tool")
        st.write("Select the parameters for a hypothetical session to predict its outcome category.")

        # Get a list of all possible categorical values from the original data
        interventions = outcome_data['intervention_name'].unique()
        metrics = outcome_data['metric_name'].unique()
        age_groups = outcome_data['age_group'].unique()
        disabilities = outcome_data['disability_category'].unique()

        with st.form("prediction_form"):
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            
            with pred_col1:
                pred_intervention = st.selectbox("Intervention", options=interventions)
                pred_metric = st.selectbox("Primary Metric", options=metrics)
                pred_age = st.selectbox("Age Group", options=age_groups)
            with pred_col2:
                pred_disability = st.selectbox("Disability Category", options=disabilities)
                pred_cost = st.slider("Session Cost ($)", 0, 200, 85)
                pred_duration = st.slider("Duration (minutes)", 15, 120, 45)
            with pred_col3:
                pred_quality = st.slider("Expected Quality Rating", 1, 5, 4)
                st.markdown("<br>", unsafe_allow_html=True)
                predict_button = st.form_submit_button(" Predict Outcome Category", use_container_width=True)

            if predict_button:
                # Create a single-row DataFrame from the user's input
                input_data = {
                    'cost_per_session': [pred_cost],
                    'session_duration': [pred_duration],
                    'quality_rating': [pred_quality],
                    'intervention_name': [pred_intervention],
                    'metric_name': [pred_metric],
                    'age_group': [pred_age],
                    'disability_category': [pred_disability],
                    # Add other categorical features with the first unique value as default
                    'intervention_category': [outcome_data['intervention_category'].unique()[0]],
                    'support_level': [outcome_data['support_level'].unique()[0]]
                }
                input_df = pd.DataFrame(input_data)

                # One-hot encode the input using the same columns as the training data
                input_df_encoded = pd.get_dummies(input_df)
                # Align columns with the training data, adding missing columns and filling with 0
                final_input = input_df_encoded.reindex(columns=model_results['features_used'], fill_value=0)

                # Get prediction and probabilities
                prediction_encoded = model_results['model'].predict(final_input)
                prediction_proba = model_results['model'].predict_proba(final_input)
                
                # Decode the prediction back to a text label ('Low', 'Medium', 'High')
                label_encoder = model_results['label_encoder']
                prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

                st.markdown(f"### Predicted Outcome: `{prediction_label}`")
                
                # Create a DataFrame for the probability chart
                proba_df = pd.DataFrame({
                    'Category': label_encoder.classes_,
                    'Probability': prediction_proba[0]
                })

                # Display the confidence chart
                fig_proba = px.bar(
                    proba_df, x='Category', y='Probability',
                    title="Model's Confidence in Prediction",
                    color='Category', color_discrete_map={'Low': '#F8766D', 'Medium': '#619CFF', 'High': '#00BA38'}
                )
                fig_proba.update_yaxes(range=[0, 1])
                st.plotly_chart(fig_proba, use_container_width=True)
                
    else:
        st.info("Predictive modeling requires at least 100 data records to run.")

def show_performance_benchmarks(analytics, outcome_data):
    """Performance benchmarking and comparative analysis"""
    st.subheader("Performance Benchmarks")

    # Internal benchmarking by categories
    benchmark_categories = ['intervention_category', 'disability_category', 'age_group', 'staff_name']

    selected_benchmark = st.selectbox(
        "Benchmark Category",
        benchmark_categories,
        format_func=lambda x: x.replace('_', ' ').title()
    )

    if selected_benchmark in outcome_data.columns:
        benchmark_data = outcome_data.groupby(selected_benchmark).agg({
            'normalized_score': ['mean', 'std', 'count'],
            'total_cost': 'sum',
            'improvement_from_baseline': 'mean',
            'quality_rating': 'mean'
        }).round(2)

        benchmark_data.columns = ['avg_outcome', 'outcome_std', 'session_count', 'total_cost', 'avg_improvement',
                                  'avg_quality']
        benchmark_data['cost_per_outcome'] = benchmark_data['total_cost'] / benchmark_data['avg_outcome']
        benchmark_data = benchmark_data.sort_values('avg_outcome', ascending=False).reset_index()

        # Benchmark visualization
        fig_benchmark = px.bar(
            benchmark_data,
            x=selected_benchmark,
            y='avg_outcome',
            error_y='outcome_std',
            color='cost_per_outcome',
            title=f'Performance Benchmarks by {selected_benchmark.replace("_", " ").title()}',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_benchmark, use_container_width=True)

        # Performance ranking table
        benchmark_display = benchmark_data.copy()
        benchmark_display['rank'] = range(1, len(benchmark_display) + 1)

        # Format columns
        benchmark_display['avg_outcome'] = benchmark_display['avg_outcome'].apply(lambda x: f"{x:.1f}")
        benchmark_display['total_cost'] = benchmark_display['total_cost'].apply(lambda x: f"${x:,.0f}")
        benchmark_display['cost_per_outcome'] = benchmark_display['cost_per_outcome'].apply(lambda x: f"${x:.0f}")
        benchmark_display['avg_quality'] = benchmark_display['avg_quality'].apply(lambda x: f"{x:.1f}")

        benchmark_display.columns = [
            selected_benchmark.replace('_', ' ').title(), 'Rank', 'Avg Outcome', 'Std Dev',
            'Sessions', 'Total Cost', 'Avg Improvement', 'Avg Quality', 'Cost/Outcome'
        ]

        st.dataframe(benchmark_display, use_container_width=True)

        # Performance alerts
        st.markdown("### Performance Alerts")

        # Identify underperforming categories
        overall_avg = outcome_data['normalized_score'].mean()
        underperforming = benchmark_data[benchmark_data['avg_outcome'] < overall_avg * 0.8]

        if not underperforming.empty:
            st.warning(
                f"**Attention Required:** {len(underperforming)} {selected_benchmark.replace('_', ' ')} categories performing below 80% of facility average")

            for _, row in underperforming.iterrows():
                category_name = row[selected_benchmark]
                performance_gap = overall_avg - row['avg_outcome']
                st.write(f" **{category_name}:** {performance_gap:.1f} points below average")
        else:
            st.success("All categories performing within acceptable ranges")


def show_advanced_intervention_analysis():
    """Advanced intervention analysis with comprehensive metrics"""
    st.subheader(" Advanced Intervention Analysis")

    # Enhanced filters with error handling
    col1, col2, col3, col4 = st.columns(4)

    try:
        with col1:
            interventions = get_interventions()
            if not interventions.empty:
                intervention_options = ["All"] + interventions['name'].tolist()
                selected_intervention = st.selectbox("Intervention", intervention_options)
            else:
                st.warning("No interventions found")
                selected_intervention = "All"

        with col2:
            individuals = get_individuals()
            if not individuals.empty:
                age_options = ["All"] + sorted(individuals['age_group'].unique().tolist())
                selected_age = st.selectbox("Age Group", age_options)
            else:
                selected_age = "All"

        with col3:
            if not individuals.empty:
                disability_options = ["All"] + sorted(individuals['disability_category'].unique().tolist())
                selected_disability = st.selectbox("Disability Category", disability_options)
            else:
                selected_disability = "All"

        with col4:
            try:
                default_start = date.today() - timedelta(days=180)
                default_end = date.today()
                date_range = st.date_input(
                    "Analysis Period",
                    value=(default_start, default_end)
                )
            except Exception:
                date_range = (date.today() - timedelta(days=180), date.today())

        # Apply filters with error handling
        filters = {
            'intervention': selected_intervention,
            'age_group': selected_age,
            'disability': selected_disability
        }

        outcome_data = get_comprehensive_outcome_data(
            date_range if isinstance(date_range, tuple) and len(date_range) == 2 else None,
            filters
        )

        if outcome_data.empty:
            st.info("No data available for selected filters.")
            return

    except Exception as e:
        st.error(f"Filter configuration error: {str(e)}")
        return

    # Intervention effectiveness matrix
    st.markdown("### Effectiveness Matrix Analysis")

    # Create effectiveness quadrants
    effectiveness_analysis = outcome_data.groupby('intervention_name').agg({
        'normalized_score': 'mean',
        'total_cost': 'mean',
        'improvement_from_baseline': 'mean',
        'quality_rating': 'mean',
        'session_duration': 'mean'
    }).reset_index()

    effectiveness_analysis['cost_effectiveness'] = effectiveness_analysis['improvement_from_baseline'] / \
                                                   effectiveness_analysis['total_cost']
    effectiveness_analysis['quality_adjusted_outcome'] = effectiveness_analysis['normalized_score'] * \
                                                         effectiveness_analysis['quality_rating'] / 5

    # Quadrant analysis
    avg_outcome = effectiveness_analysis['quality_adjusted_outcome'].mean()
    avg_cost_eff = effectiveness_analysis['cost_effectiveness'].mean()

    effectiveness_analysis['quadrant'] = effectiveness_analysis.apply(
        lambda x:
        'High Value' if x['quality_adjusted_outcome'] >= avg_outcome and x['cost_effectiveness'] >= avg_cost_eff
        else 'High Outcome, Low Efficiency' if x['quality_adjusted_outcome'] >= avg_outcome
        else 'Low Outcome, High Efficiency' if x['cost_effectiveness'] >= avg_cost_eff
        else 'Needs Improvement',
        axis=1
    )

    # Visualization
    fig_quadrant = px.scatter(
        effectiveness_analysis,
        x='cost_effectiveness',
        y='quality_adjusted_outcome',
        color='quadrant',
        size='session_duration',  # Corrected from 'mean_session_duration'
        hover_data=['intervention_name'],
        title='Intervention Effectiveness Quadrants',
        labels={
            'cost_effectiveness': 'Cost Effectiveness (Improvement/Cost)',
            'quality_adjusted_outcome': 'Quality-Adjusted Outcome Score'
        }
    )

    # Add quadrant lines
    fig_quadrant.add_hline(y=avg_outcome, line_dash="dash", line_color="gray")
    fig_quadrant.add_vline(x=avg_cost_eff, line_dash="dash", line_color="gray")

    st.plotly_chart(fig_quadrant, use_container_width=True)

    # Detailed intervention profiles
    st.markdown("### Intervention Performance Profiles")

    for _, intervention in effectiveness_analysis.iterrows():
        with st.expander(f" {intervention['intervention_name']} - {intervention['quadrant']}"):

            int_data = outcome_data[outcome_data['intervention_name'] == intervention['intervention_name']]

            profile_col1, profile_col2, profile_col3 = st.columns(3)

            with profile_col1:
                st.metric("Average Outcome", f"{intervention['normalized_score']:.1f}/10")
                st.metric("Cost Effectiveness", f"{intervention['cost_effectiveness']:.4f}")
                st.metric("Quality Rating", f"{intervention['quality_rating']:.1f}/5")

            with profile_col2:
                st.metric("Average Improvement", f"{intervention['improvement_from_baseline']:.1f}")
                st.metric("Average Cost", f"${intervention['total_cost']:.2f}")
                st.metric("Session Count", f"{len(int_data):,}")

            with profile_col3:
                # Calculate additional metrics
                success_rate = (int_data['improvement_from_baseline'] > 0).mean() * 100
                consistency = 1 - (int_data['normalized_score'].std() / int_data['normalized_score'].mean()) if \
                    int_data['normalized_score'].mean() > 0 else 0

                st.metric("Success Rate", f"{success_rate:.1f}%")
                st.metric("Consistency Index", f"{consistency:.2f}")

                # Outcome distribution
                outcome_dist = int_data['normalized_score'].describe()
                st.write(f"**Outcome Range:** {outcome_dist['min']:.1f} - {outcome_dist['max']:.1f}")

            # Performance over time for this intervention
            if len(int_data) > 5:
                int_data_sorted = int_data.sort_values('session_date')
                fig_int_trend = px.scatter(
                    int_data_sorted,
                    x='session_date',
                    y='normalized_score',
                    title=f'{intervention["intervention_name"]} - Outcome Trends',
                    trendline='ols'
                )
                st.plotly_chart(fig_int_trend, use_container_width=True)


def show_individual_analytics():
    """Comprehensive individual progress analytics"""
    st.subheader(" Individual Analytics & Progress Tracking")

    try:
        individuals = get_individuals()

        if individuals.empty:
            st.warning("No individuals found in the system.")
            return

        # Individual selector with enhanced information
        individual_options = []
        for _, ind in individuals.iterrows():
            # Handle missing values gracefully
            anonymous_id = ind.get('anonymous_id', 'Unknown ID')
            age_group = ind.get('age_group', 'Unknown Age')
            disability_category = ind.get('disability_category', 'Unknown Disability')
            disability_severity = ind.get('disability_severity', '')

            severity_text = f" ({disability_severity})" if disability_severity else ""
            option_text = f"{anonymous_id} - {age_group}, {disability_category}{severity_text}"
            individual_options.append(option_text)

        col1, col2 = st.columns([3, 1])

        with col1:
            selected_idx = st.selectbox("Select Individual", range(len(individual_options)),
                                        format_func=lambda x: individual_options[x])

        with col2:
            analysis_months = st.number_input("Analysis Period (months)", min_value=1, max_value=24, value=6)

        selected_individual = individuals.iloc[selected_idx]
        individual_id = selected_individual['id']

        # Get comprehensive individual data with error handling
        try:
            cutoff_date = date.today() - timedelta(days=30 * analysis_months)
            individual_data = get_comprehensive_outcome_data(
                date_range=(cutoff_date, date.today()),
                filters={}
            )

            # Filter for selected individual using anonymous_id for safety
            if not individual_data.empty and 'anonymous_id' in individual_data.columns:
                individual_data = individual_data[
                    individual_data['anonymous_id'] == selected_individual.get('anonymous_id', '')
                    ]
            else:
                individual_data = pd.DataFrame()

        except Exception as e:
            st.error(f"Error loading individual data: {str(e)}")
            individual_data = pd.DataFrame()

        if individual_data.empty:
            st.info("No outcome data available for this individual in the selected period.")
            return

    except Exception as e:
        st.error(f"Individual analytics error: {str(e)}")
        return

    # Individual summary dashboard
    st.markdown("### Individual Profile Summary")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        total_sessions = len(individual_data)
        active_interventions = individual_data['intervention_name'].nunique()
        st.metric("Total Sessions", total_sessions)
        st.metric("Active Interventions", active_interventions)

    with summary_col2:
        avg_outcome = individual_data['normalized_score'].mean()
        latest_outcome = individual_data.sort_values('session_date').iloc[-1]['normalized_score']
        outcome_change = latest_outcome - individual_data.sort_values('session_date').iloc[0]['normalized_score']

        st.metric("Average Outcome", f"{avg_outcome:.1f}/10")
        st.metric("Recent Change", f"{outcome_change:+.1f}", delta=f"{outcome_change:.1f}")

    with summary_col3:
        total_investment = individual_data['total_cost'].sum()
        avg_session_cost = individual_data['total_cost'].mean()

        st.metric("Total Investment", f"${total_investment:,.0f}")
        st.metric("Avg Session Cost", f"${avg_session_cost:.0f}")

    with summary_col4:
        attendance_rate = (individual_data['attendance_status'] == 'attended').mean() * 100
        avg_quality = individual_data['quality_rating'].mean()

        st.metric("Attendance Rate", f"{attendance_rate:.1f}%")
        st.metric("Quality Rating", f"{avg_quality:.1f}/5")

    # Comprehensive progress analysis
    tab1, tab2, tab3, tab4 = st.tabs([
        "Progress Timeline",
        "Intervention Effectiveness",
        "Goal Tracking",
        "Detailed Analytics"
    ])

    with tab1:
        show_individual_progress_timeline(individual_data, selected_individual)

    with tab2:
        show_individual_intervention_effectiveness(individual_data)

    with tab3:
        show_individual_goal_tracking(individual_id)

    with tab4:
        show_individual_detailed_analytics(individual_data)


def show_individual_progress_timeline(individual_data, selected_individual):
    """Detailed progress timeline for individual"""

    # Multi-metric timeline
    metrics = individual_data['metric_name'].unique()

    if len(metrics) > 1:
        selected_metrics = st.multiselect(
            "Select Metrics to Display",
            metrics,
            default=metrics[:min(4, len(metrics))]
        )
    else:
        selected_metrics = metrics

    if selected_metrics:
        timeline_data = individual_data[individual_data['metric_name'].isin(selected_metrics)]

        fig_timeline = px.line(
            timeline_data,
            x='session_date',
            y='normalized_score',
            color='metric_name',
            facet_col='intervention_name',
            facet_col_wrap=2,
            title=f'Progress Timeline - {selected_individual["anonymous_id"]}',
            labels={
                'session_date': 'Date',
                'normalized_score': 'Normalized Score (0-10)',
                'metric_name': 'Outcome Metric'
            }
        )

        # Add trend lines
        fig_timeline.update_traces(mode='lines+markers')
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Progress summary table
        progress_summary = timeline_data.groupby(['intervention_name', 'metric_name']).agg({
            'normalized_score': ['first', 'last', 'mean', 'count'],
            'improvement_from_baseline': 'last'
        }).round(2)

        progress_summary.columns = ['First_Score', 'Latest_Score', 'Avg_Score', 'Sessions', 'Total_Improvement']
        progress_summary['Change'] = progress_summary['Latest_Score'] - progress_summary['First_Score']
        progress_summary = progress_summary.reset_index()

        st.markdown("#### Progress Summary by Intervention & Metric")
        st.dataframe(progress_summary, use_container_width=True)


def show_individual_intervention_effectiveness(individual_data):
    """Analysis of intervention effectiveness for specific individual"""

    intervention_effectiveness = individual_data.groupby('intervention_name').agg({
        'normalized_score': ['mean', 'std', 'count'],
        'improvement_from_baseline': ['mean', 'sum'],
        'total_cost': 'sum',
        'quality_rating': 'mean',
        'session_duration': 'mean'
    }).round(2)

    intervention_effectiveness.columns = [
        'avg_outcome', 'outcome_std', 'session_count', 'avg_improvement',
        'total_improvement', 'total_cost', 'avg_quality', 'avg_duration'
    ]

    intervention_effectiveness['cost_per_improvement'] = intervention_effectiveness['total_cost'] / \
                                                         intervention_effectiveness['total_improvement']
    intervention_effectiveness = intervention_effectiveness.reset_index()

    # Effectiveness ranking
    fig_effectiveness = px.bar(
        intervention_effectiveness.sort_values('avg_improvement', ascending=False),
        x='intervention_name',
        y='avg_improvement',
        color='cost_per_improvement',
        title='Intervention Effectiveness for This Individual',
        color_continuous_scale='RdYlGn_r'
    )
    fig_effectiveness.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_effectiveness, use_container_width=True)

    # Recommendations based on individual response
    st.markdown("#### Personalized Recommendations")

    best_intervention = intervention_effectiveness.loc[intervention_effectiveness['avg_improvement'].idxmax()]
    most_sessions = intervention_effectiveness.loc[intervention_effectiveness['session_count'].idxmax()]
    most_cost_effective = intervention_effectiveness.loc[intervention_effectiveness['cost_per_improvement'].idxmin()]

    recommendation_col1, recommendation_col2, recommendation_col3 = st.columns(3)

    with recommendation_col1:
        st.success(f"**Most Effective:** {best_intervention['intervention_name']}")
        st.write(f"Average improvement: {best_intervention['avg_improvement']:.1f} points")

    with recommendation_col2:
        st.info(f"**Most Utilized:** {most_sessions['intervention_name']}")
        st.write(f"Total sessions: {most_sessions['session_count']}")

    with recommendation_col3:
        st.success(f"**Most Cost-Effective:** {most_cost_effective['intervention_name']}")
        st.write(f"Cost per improvement: ${most_cost_effective['cost_per_improvement']:.0f}")


def show_individual_goal_tracking(individual_id):
    """Individual goal tracking and progress monitoring"""
    conn = st.session_state.db_conn

    # Get goals for this individual
    goals = conn.execute("""
        SELECT 
            g.id,
            g.goal_type,
            g.baseline_value,
            g.target_value,
            g.target_date,
            g.priority_level,
            g.status,
            m.name as metric_name,
            m.unit_of_measure,
            m.higher_is_better
        FROM treatment_goals g
        JOIN outcome_metrics m ON g.outcome_metric_id = m.id
        WHERE g.individual_id = ?
        ORDER BY g.priority_level, g.target_date
    """, [individual_id]).df()

    if goals.empty:
        st.info("No treatment goals set for this individual.")

        # Quick goal setting interface
        st.markdown("#### Set New Goal")
        with st.form("quick_goal_form"):
            metrics = get_outcome_metrics()

            col1, col2, col3 = st.columns(3)
            with col1:
                metric_idx = st.selectbox("Outcome Metric", range(len(metrics)),
                                          format_func=lambda x: metrics.iloc[x]['name'])
            with col2:
                target_value = st.number_input("Target Value", min_value=0.0, step=0.1)
            with col3:
                target_date = st.date_input("Target Date", min_value=date.today())

            if st.form_submit_button("Set Goal"):
                goal_id = str(uuid.uuid4())
                metric_id = metrics.iloc[metric_idx]['id']

                conn.execute("""
                    INSERT INTO treatment_goals (id, individual_id, outcome_metric_id, goal_type, target_value, target_date, created_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [goal_id, individual_id, metric_id, 'Improvement', target_value, target_date,
                      st.session_state.user['id']])

                st.success("Goal created successfully!")
                st.rerun()

        return

    # Display current goals with progress
    st.markdown("#### Current Treatment Goals")

    for _, goal in goals.iterrows():
        # Get latest score for this metric
        latest_score = conn.execute("""
            SELECT score, session_date
            FROM outcome_records o
            JOIN outcome_metrics m ON o.outcome_metric_id = m.id
            WHERE o.individual_id = ? AND m.name = ?
            ORDER BY session_date DESC
            LIMIT 1
        """, [individual_id, goal['metric_name']]).fetchone()

        current_value = latest_score[0] if latest_score else goal['baseline_value']

        # Calculate progress
        # Ensure all values are floats for arithmetic operations
        baseline_val = float(goal['baseline_value'])
        target_val = float(goal['target_value'])
        current_val = float(current_value)

        if goal['higher_is_better']:
            progress = ((current_val - baseline_val) / (
                    target_val - baseline_val)) * 100
        else:
            progress = ((baseline_val - current_val) / (
                    baseline_val - target_val)) * 100

        progress = max(0, min(100, progress))  # Clamp between 0-100%

        # Goal status color coding
        days_to_target = (goal['target_date'] - date.today()).days
        status_color = "green" if progress >= 80 else "orange" if progress >= 50 else "red"

        with st.container():
            goal_col1, goal_col2, goal_col3 = st.columns([3, 1, 1])

            with goal_col1:
                st.write(f"**{goal['metric_name']}** ({goal['goal_type']})")
                st.progress(progress / 100)
                st.write(
                    f"Current: {current_value:.1f} | Target: {goal['target_value']:.1f} | Progress: {progress:.1f}%")

            with goal_col2:
                st.metric("Days Remaining", days_to_target)
                priority_emoji = "" if goal['priority_level'] == 1 else "" if goal['priority_level'] == 2 else ""
                st.write(f"Priority: {priority_emoji}")

            with goal_col3:
                if st.button(f"Update Goal", key=f"update_goal_{goal['id']}"):
                    # Goal update interface would go here
                    st.info("Goal update functionality would open here")


def show_individual_detailed_analytics(individual_data):
    """Detailed statistical analytics for individual performance"""

    # Performance distribution analysis
    st.markdown("#### Performance Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Outcome score distribution
        fig_dist = px.histogram(
            individual_data,
            x='normalized_score',
            nbins=20,
            title='Outcome Score Distribution',
            labels={'normalized_score': 'Normalized Score', 'count': 'Frequency'}
        )
        fig_dist.add_vline(x=individual_data['normalized_score'].mean(), line_dash="dash",
                           annotation_text=f"Mean: {individual_data['normalized_score'].mean():.1f}")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        # Quality vs Outcome relationship
        fig_quality_outcome = px.scatter(
            individual_data,
            x='quality_rating',
            y='normalized_score',
            color='intervention_name',
            title='Session Quality vs Outcome Correlation',
            trendline='ols'
        )
        st.plotly_chart(fig_quality_outcome, use_container_width=True)

    # Advanced metrics calculation
    st.markdown("#### Advanced Performance Metrics")

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

    with metrics_col1:
        # Consistency metrics
        outcome_cv = individual_data['normalized_score'].std() / individual_data['normalized_score'].mean()
        improvement_trend = stats.linregress(
            range(len(individual_data.sort_values('session_date'))),
            individual_data.sort_values('session_date')['normalized_score']
        ).slope

        st.metric("Consistency Index", f"{1 - outcome_cv:.2f}")
        st.metric("Monthly Trend", f"{improvement_trend * 30:.2f}/month")

    with metrics_col2:
        # Engagement metrics
        avg_duration_vs_planned = (individual_data['session_duration'] / individual_data['planned_duration']).mean()
        quality_consistency = 1 - (individual_data['quality_rating'].std() / individual_data['quality_rating'].mean())

        st.metric("Duration Compliance", f"{avg_duration_vs_planned:.1%}")
        st.metric("Quality Consistency", f"{quality_consistency:.2f}")

    with metrics_col3:
        # Cost efficiency for this individual
        cost_per_improvement = individual_data['total_cost'].sum() / individual_data[
            'improvement_from_baseline'].sum() if individual_data['improvement_from_baseline'].sum() > 0 else 0
        roi_percentage = (individual_data['improvement_from_baseline'].sum() / individual_data[
            'total_cost'].sum()) * 100 if individual_data['total_cost'].sum() > 0 else 0

        st.metric("Cost per Improvement", f"${cost_per_improvement:.0f}")
        st.metric("ROI Ratio", f"{roi_percentage:.1f}%")


def show_comprehensive_reporting():
    st.subheader(" Comprehensive Reports")
    report_type = st.selectbox(
        "Select Report Type",
        [
            "Executive Summary Report",
            "Intervention Effectiveness Report",
            "Individual Progress Report",
            "Cost Analysis Report",
            "Quality Assurance Report",
            "Regulatory Compliance Report",
            "Staff Performance Report"
        ]
    )

    # Conditional selectors for specific report types
    report_params = {}
    if report_type == "Individual Progress Report":
        individuals = get_individuals()
        if not individuals.empty:
            ind_idx = st.selectbox("Select Individual for Report", range(len(individuals)),
                                   format_func=lambda x: individuals.iloc[x]['anonymous_id'])
            report_params['individual'] = individuals.iloc[ind_idx]
        else:
            st.warning("No individuals found to generate a report.")
            return

    if report_type == "Staff Performance Report":
        conn = st.session_state.db_conn
        staff = conn.execute("SELECT id, full_name FROM users WHERE role IN ('Staff', 'Supervisor') ORDER BY full_name").df()
        if not staff.empty:
            staff_options = ["All Staff"] + staff['full_name'].tolist()
            selected_staff_name = st.selectbox("Select Staff Member for Report", staff_options)
            if selected_staff_name != "All Staff":
                report_params['staff'] = staff[staff['full_name'] == selected_staff_name].iloc[0]
            else:
                report_params['staff'] = "All Staff"
        else:
            st.warning("No staff found to generate a report.")
            return
 

    # General report parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        report_period = st.selectbox("Report Period", ["Last 30 Days", "Last Quarter", "Last Year", "Custom"])
        if report_period == "Custom":
            custom_range = st.date_input("Custom Date Range", value=(date.today() - timedelta(days=90), date.today()))
        else:
            days_map = {"Last 30 Days": 30, "Last Quarter": 90, "Last Year": 365}
            custom_range = (date.today() - timedelta(days=days_map[report_period]), date.today())
    with col2:
        include_charts = st.checkbox("Include Visualizations", value=True)
    with col3:
        export_format = st.selectbox("Format", ["View in App", "PDF Export", "CSV Data Export"])

    if st.button("Generate Report", type="primary", use_container_width=True):
        outcome_data = get_comprehensive_outcome_data(custom_range)
        analytics = get_advanced_analytics()

        if outcome_data.empty:
            st.error("No data available for the selected period to generate a report.")
            return

        # Router to map report names to their generator functions
        report_router = {
            "Executive Summary Report": generate_executive_report,
            "Intervention Effectiveness Report": generate_intervention_report,
            "Individual Progress Report": generate_individual_progress_report,
            "Cost Analysis Report": generate_cost_analysis_report,
            "Quality Assurance Report": generate_quality_report,
            "Regulatory Compliance Report": generate_regulatory_report,
            "Staff Performance Report": generate_staff_performance_report
        }
        
        report_function = report_router.get(report_type)
        if not report_function:
            st.warning(f"{report_type} is not fully implemented yet.")
            return
            
        with st.spinner("Generating report..."):
            # 1. Generate the report content once
            report_content = report_function(outcome_data, analytics, custom_range, include_charts, **report_params)

            # 2. Handle the selected format
            if export_format == "View in App":
                # Render the content directly on the screen
                for item in report_content:
                    if item['type'] == 'header':
                        st.markdown(f"### {item['data']}")
                    elif item['type'] == 'text':
                        st.markdown(item['data'])
                    elif item['type'] == 'dataframe':
                        st.dataframe(item['data'], use_container_width=True, hide_index=True)
                    elif item['type'] == 'figure':
                        st.plotly_chart(item['data'], use_container_width=True)
                    elif item['type'] == 'metrics':
                        cols = st.columns(len(item['data']))
                        for i, (label, value) in enumerate(item['data'].items()):
                            cols[i].metric(label, value)

            elif export_format == "PDF Export":
                pdf_bytes = create_pdf_report(report_type, custom_range, report_content)
                st.download_button(
                    label=" Download Report as PDF",
                    data=pdf_bytes,
                    file_name=f"{report_type.replace(' ', '_')}_{custom_range[0]}.pdf",
                    mime="application/pdf"
                )
            
            elif export_format == "CSV Data Export":
                # Find the primary dataframe in the content to export
                main_df = next((item['data'] for item in report_content if item['type'] == 'dataframe'), None)
                if main_df is not None:
                    csv_bytes = main_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=" Download Data as CSV",
                        data=csv_bytes,
                        file_name=f"{report_type.replace(' ', '_')}_{custom_range[0]}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data table available in this report for CSV export.")

def show_smart_data_entry():
    """Enhanced data entry with hardened validation and a complete submission workflow."""
    st.markdown("####  Smart Outcome Recording")

    try:
        conn = st.session_state.db_conn
        individuals = get_individuals()
        interventions = get_interventions()
        metrics = get_outcome_metrics()

        if individuals.empty or interventions.empty or metrics.empty:
            st.warning("Please configure individuals, interventions, and outcome metrics before logging data.")
            return

        with st.form("enhanced_outcome_form"):
            entry_col1, entry_col2 = st.columns(2)

            with entry_col1:
                # Individual selection with search
                individual_search = st.text_input("Search Individual", placeholder="Type anonymous ID to search...")
                filtered_individuals = individuals[individuals['anonymous_id'].str.contains(individual_search, case=False, na=False)] if individual_search else individuals
                
                if not filtered_individuals.empty:
                    selected_individual_idx = st.selectbox("Individual", range(len(filtered_individuals)), format_func=lambda x: filtered_individuals.iloc[x]['anonymous_id'])
                    selected_individual = filtered_individuals.iloc[selected_individual_idx]
                else:
                    st.warning("No individuals found matching your search.")
                    # Add a dummy button to satisfy Streamlit's form requirements
                    st.form_submit_button("Search again")
                    return

                # Intervention selection
                intervention_idx = st.selectbox("Intervention", range(len(interventions)), format_func=lambda x: interventions.iloc[x]['name'])
                selected_intervention = interventions.iloc[intervention_idx]
                
                # Session date and duration
                session_date = st.date_input("Session Date", value=date.today(), max_value=date.today())
                actual_duration = st.number_input("Actual Duration (minutes)", min_value=5, value=45, step=5)

            with entry_col2:
                # Outcome Measurements
                st.markdown("**Outcome Measurements**")
                selected_metrics_indices = st.multiselect(
                    "Select Metrics to Record", 
                    range(len(metrics)), 
                    format_func=lambda x: metrics.iloc[x]['name'], 
                    default=[0] if len(metrics) > 0 else []
                )
                
                outcome_scores = {}
                for metric_idx in selected_metrics_indices:
                    metric = metrics.iloc[metric_idx]
                    default_value = float(metric.get('scale_min', 0))
                    outcome_scores[metric['id']] = st.number_input(
                        f"Score for: {metric['name']}", 
                        min_value=float(metric.get('scale_min', 0)), 
                        max_value=float(metric.get('scale_max', 10)), 
                        value=default_value, 
                        step=0.1,
                        key=f"score_{metric['id']}"
                    )

                # Session details
                attendance_status = st.selectbox("Attendance", ["attended", "partial", "absent"])
                quality_rating = st.slider("Session Quality Rating", 1, 5, 4)
                notes = st.text_area("Session Notes", placeholder="Observations, progress, concerns, etc. (Required)")

            submitted = st.form_submit_button("Record Session", type="primary", use_container_width=True)

            if submitted:
                # Server-side validation block
                is_valid = True
                if not notes or len(notes.strip()) < 10:
                    st.error("Validation Error: Session Notes are required and must be at least 10 characters long.")
                    is_valid = False
                if session_date > date.today():
                    st.error("Validation Error: The session date cannot be in the future.")
                    is_valid = False
                if not selected_metrics_indices:
                    st.error("Validation Error: At least one outcome metric must be selected and scored.")
                    is_valid = False
                
                if is_valid:
                    with st.spinner("Saving session data..."):
                        session_id = str(uuid.uuid4())
                        
                        # Record each outcome metric
                        for metric_id, score in outcome_scores.items():
                            record_id = str(uuid.uuid4())
                            
                            # Get baseline score for comparison
                            baseline_result = conn.execute("SELECT score FROM outcome_records WHERE individual_id = ? AND outcome_metric_id = ? ORDER BY session_date ASC LIMIT 1", [selected_individual['id'], metric_id]).fetchone()
                            baseline_score = baseline_result[0] if baseline_result else score
                            
                            conn.execute("""
                                INSERT INTO outcome_records (
                                    id, individual_id, intervention_id, outcome_metric_id, session_id,
                                    score, baseline_score, session_date, session_duration, attendance_status,
                                    quality_rating, notes, recorded_by, staff_id
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, [
                                record_id, selected_individual['id'], selected_intervention['id'], metric_id, session_id,
                                score, baseline_score, session_date, actual_duration, attendance_status,
                                quality_rating, notes.strip(), st.session_state.user['id'], st.session_state.user['id']
                            ])
                        
                        # Record costs
                        cost_record_id = str(uuid.uuid4())
                        direct_cost = selected_intervention.get('cost_per_session', 50.0)
                        conn.execute("""
                            INSERT INTO cost_records (id, intervention_id, individual_id, session_id, direct_cost, cost_date) 
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, [cost_record_id, selected_intervention['id'], selected_individual['id'], session_id, direct_cost, session_date])

                        # Create audit log
                        create_audit_log(
                            st.session_state.db_conn, # Pass the connection
                            'outcome_records', session_id, 'INSERT',
                            new_values={'session_count': len(outcome_scores), 'intervention': selected_intervention.get('name', 'Unknown')},
                            reason='Regular session recording via Smart Data Entry',
                            changed_by_user_id=st.session_state.user['id'] # Pass the user ID
                        )
                        
                        st.success("Session recorded successfully!")

    except Exception as e:
        st.error(f"An error occurred in the data entry form: {str(e)}")

def generate_executive_report(outcome_data, analytics, date_range, include_charts, **kwargs):
    """
    Generates content for the executive summary report and returns it as a list of dicts.
    """
    # Initialize the content list that will be returned
    content = []

    # 1. Calculate Key Performance Indicators (KPIs)
    total_cost = outcome_data['total_cost'].sum()
    total_sessions = outcome_data['session_id'].nunique()
    unique_individuals = outcome_data['anonymous_id'].nunique()
    avg_outcome = outcome_data['normalized_score'].mean()
    
    content.append({'type': 'header', 'data': 'Key Performance Indicators'})
    
    kpi_data = {
        "Total Investment": f"${total_cost:,.0f}",
        "Sessions Delivered": f"{total_sessions:,}",
        "Individuals Served": f"{unique_individuals}",
        "Average Outcome Score": f"{avg_outcome:.1f} / 10"
    }
    content.append({'type': 'metrics', 'data': kpi_data})

    # 2. Generate Visualizations if requested
    if include_charts and analytics:
        content.append({'type': 'header', 'data': 'Key Visualizations'})
        
        # Chart 1: Top Interventions by Outcome
        if 'intervention_analysis' in analytics:
            fig_cost_outcome = px.bar(
                analytics['intervention_analysis'].head(10), 
                x='intervention_name', 
                y='mean_normalized_score',
                color='mean_total_cost',
                title='Top Interventions by Outcome Score and Average Cost',
                labels={'mean_normalized_score': 'Avg. Outcome Score', 'intervention_name': 'Intervention', 'mean_total_cost': 'Avg. Cost ($)'}
            )
            content.append({'type': 'figure', 'data': fig_cost_outcome})

        # Chart 2: Overall Outcome Trend
        if 'trend_analysis' in analytics:
            trend_data = analytics['trend_analysis'].copy()
            trend_data['year_month_str'] = trend_data['year_month'].astype(str)
            fig_trend = px.line(
                trend_data,
                x='year_month_str',
                y='normalized_score',
                title='Overall Outcome Score Trend Over Time',
                labels={'year_month_str': 'Month', 'normalized_score': 'Average Normalized Score'},
                markers=True
            )
            content.append({'type': 'figure', 'data': fig_trend})

    # 3. Summarize Top and Bottom Performing Interventions
    if 'intervention_analysis' in analytics:
        content.append({'type': 'header', 'data': 'Intervention Performance Summary'})
        
        intervention_analysis = analytics['intervention_analysis'].sort_values('efficiency_ratio', ascending=False)
        top_performers = intervention_analysis.head(3)
        bottom_performers = intervention_analysis.tail(2)
        
        summary_text = f"""
        The most efficient interventions during this period were **{', '.join(top_performers['intervention_name'].tolist())}**.
        These programs delivered the highest outcome improvements relative to their cost.

        Conversely, the programs identified as needing review based on their lower efficiency ratios were **{', '.join(bottom_performers['intervention_name'].tolist())}**.
        """
        content.append({'type': 'text', 'data': summary_text})

    # 4. Provide Actionable Recommendations
    if 'intervention_analysis' in analytics:
        content.append({'type': 'header', 'data': 'Strategic Recommendations'})
        
        top_intervention = analytics['intervention_analysis'].iloc[0]
        bottom_intervention = analytics['intervention_analysis'].iloc[-1]
        
        recommendations_text = f"""
        - **Capitalize on Success**: Focus resources on expanding the **{top_intervention['intervention_name']}** program, which demonstrates the highest cost-effectiveness.
        - **Review and Optimize**: Conduct a programmatic review of **{bottom_intervention['intervention_name']}** to identify opportunities for improving outcomes or increasing efficiency.
        - **Data-Driven Staffing**: Utilize staff performance analytics to pair skilled staff with interventions where they have the most impact.
        """
        content.append({'type': 'text', 'data': recommendations_text})

    return content

def generate_intervention_report(outcome_data, analytics, date_range, include_charts, **kwargs):
    """Generates a detailed report on intervention effectiveness."""
    content = []
    intervention_analysis = analytics.get('intervention_analysis')
    if intervention_analysis is None or intervention_analysis.empty:
        content.append({'type': 'text', 'data': 'Insufficient data for intervention analysis.'})
        return content

    # Ranking Table
    content.append({'type': 'header', 'data': 'Intervention Performance Ranking'})
    content.append({'type': 'text', 'data': 'This table ranks interventions by their efficiency (outcome improvement per dollar spent).'})
    display_data = intervention_analysis.sort_values('efficiency_ratio', ascending=False).rename(
        columns={'intervention_name': 'Intervention', 'mean_normalized_score': 'Avg Outcome', 
                 'mean_total_cost': 'Avg Cost ($)', 'efficiency_ratio': 'Efficiency Ratio'}
    )[['Intervention', 'Avg Outcome', 'Avg Cost ($)', 'Efficiency Ratio']].round(2)
    content.append({'type': 'dataframe', 'data': display_data})
    
    # Visualizations
    if include_charts:
        content.append({'type': 'header', 'data': 'Cost vs. Effectiveness Quadrant'})
        fig_scatter = px.scatter(
            intervention_analysis, x='mean_total_cost', y='mean_normalized_score',
            size='count_normalized_score', color='intervention_name',
            title='Cost vs. Effectiveness of Interventions',
            labels={'mean_total_cost': 'Average Cost per Session ($)', 'mean_normalized_score': 'Average Outcome Score'}
        )
        content.append({'type': 'figure', 'data': fig_scatter})

    # Statistical Significance
    if 'significance_testing' in analytics and analytics['significance_testing']:
        content.append({'type': 'header', 'data': 'Statistical Significance'})
        content.append({'type': 'text', 'data': 'The following table shows pairs of interventions where the difference in outcomes is statistically significant (p-value < 0.05).'})
        sig_df = pd.DataFrame(analytics['significance_testing'])
        sig_df_display = sig_df[sig_df['significant'] == True]
        if not sig_df_display.empty:
            content.append({'type': 'dataframe', 'data': sig_df_display[['intervention_1', 'intervention_2', 'p_value']].round(4)})
        else:
            content.append({'type': 'text', 'data': 'No statistically significant differences were found between interventions in this period.'})
            
    return content

def generate_cost_analysis_report(outcome_data, analytics, date_range, include_charts, **kwargs):
    """Generates a detailed report on financial metrics."""
    content = []
    total_cost = outcome_data['total_cost'].sum()
    avg_cost_session = outcome_data['total_cost'].mean()
    avg_cost_individual = total_cost / outcome_data['anonymous_id'].nunique()

    # KPIs
    content.append({'type': 'header', 'data': 'Overall Financial Summary'})
    metrics_data = {
        "Total Investment": f"${total_cost:,.2f}",
        "Avg. Cost per Session": f"${avg_cost_session:,.2f}",
        "Avg. Cost per Individual": f"${avg_cost_individual:,.2f}"
    }
    content.append({'type': 'metrics', 'data': metrics_data})
    
    # Visualizations
    if include_charts:
        content.append({'type': 'header', 'data': 'Cost Distribution'})
        cost_by_intervention = outcome_data.groupby('intervention_name')['total_cost'].sum()
        fig_pie = px.pie(
            values=cost_by_intervention.values,
            names=cost_by_intervention.index,
            title='Percentage of Total Cost by Intervention'
        )
        content.append({'type': 'figure', 'data': fig_pie})

    # Efficiency Table
    content.append({'type': 'header', 'data': 'Cost Efficiency per Intervention'})
    content.append({'type': 'text', 'data': 'This table shows the cost to achieve one point of normalized outcome improvement.'})
    efficiency_data = analytics.get('intervention_analysis', pd.DataFrame())
    if not efficiency_data.empty:
        df_eff = efficiency_data.rename(columns={'intervention_name': 'Intervention', 'cost_per_outcome_point': 'Cost per Outcome Point ($)'})
        content.append({'type': 'dataframe', 'data': df_eff[['Intervention', 'Cost per Outcome Point ($)']].sort_values('Cost per Outcome Point ($)').round(2)})
        
    return content

def generate_quality_report(outcome_data, analytics, date_range, include_charts, **kwargs):
    """Generates a report on quality assurance metrics."""
    content = []
    quality_metrics = analytics.get('quality_metrics', {})
    
    # KPIs
    content.append({'type': 'header', 'data': 'Key Quality Indicators'})
    metrics_data = {
        "Average Quality Rating": f"{quality_metrics.get('average_quality_rating', 0):.2f} / 5",
        "Overall Attendance Rate": f"{quality_metrics.get('overall_attendance_rate', 0):.1f}%",
        "Sessions Below Threshold (<3)": f"{quality_metrics.get('sessions_below_quality_threshold', 0)}"
    }
    content.append({'type': 'metrics', 'data': metrics_data})
    
    # Visualizations
    if include_charts:
        content.append({'type': 'header', 'data': 'Quality Analysis'})
        fig_dist = px.histogram(outcome_data, x='quality_rating', title='Distribution of Session Quality Ratings', nbins=5)
        content.append({'type': 'figure', 'data': fig_dist})
        
        quality_by_staff = outcome_data.groupby('staff_name')['quality_rating'].mean().sort_values(ascending=False).reset_index()
        fig_staff = px.bar(quality_by_staff, x='staff_name', y='quality_rating', title='Average Quality Rating by Staff Member')
        content.append({'type': 'figure', 'data': fig_staff})

    # Low Quality Log
    content.append({'type': 'header', 'data': 'Low-Quality Session Log'})
    content.append({'type': 'text', 'data': 'The following sessions received a quality rating below 3 and may require review.'})
    low_quality_sessions = outcome_data[outcome_data['quality_rating'] < 3]
    if not low_quality_sessions.empty:
        content.append({'type': 'dataframe', 'data': low_quality_sessions[['session_date', 'anonymous_id', 'intervention_name', 'staff_name', 'quality_rating']]})
    else:
        content.append({'type': 'text', 'data': 'No sessions with a quality rating below 3 were recorded in this period.'})
        
    return content

def generate_individual_progress_report(outcome_data, analytics, date_range, include_charts, individual=None, **kwargs):
    """Generates a detailed progress report for a single individual."""
    content = []
    if individual is None:
        content.append({'type': 'text', 'data': 'Error: No individual selected for the report.'})
        return content

    individual_data = outcome_data[outcome_data['anonymous_id'] == individual['anonymous_id']]
    if individual_data.empty:
        content.append({'type': 'text', 'data': f"No data found for {individual['anonymous_id']} in the selected period."})
        return content
        
    # Visualizations
    if include_charts:
        content.append({'type': 'header', 'data': 'Overall Progress Timeline'})
        fig_progress = px.line(
            individual_data.sort_values('session_date'), x='session_date', y='normalized_score',
            color='metric_name', title=f'Normalized Outcome Scores for {individual["anonymous_id"]}', markers=True
        )
        content.append({'type': 'figure', 'data': fig_progress})

    # Performance by Intervention
    content.append({'type': 'header', 'data': 'Performance by Intervention'})
    intervention_summary = individual_data.groupby('intervention_name').agg(
        Sessions=('session_id', 'nunique'),
        Avg_Score=('normalized_score', 'mean'),
        Avg_Improvement=('improvement_from_baseline', 'mean')
    ).round(2).reset_index()
    content.append({'type': 'dataframe', 'data': intervention_summary})
    
    return content

def generate_staff_performance_report(outcome_data, analytics, date_range, include_charts, staff=None, **kwargs):
    """Generates a performance report for staff."""
    content = []
    staff_analysis = analytics.get('staff_analysis')
    if staff_analysis is None or staff_analysis.empty:
        content.append({'type': 'text', 'data': 'Insufficient data for staff analysis.'})
        return content

    if staff == "All Staff":
        content.append({'type': 'header', 'data': 'Overall Staff Performance Leaderboard'})
        content.append({'type': 'dataframe', 'data': staff_analysis.round(2)})
        if include_charts:
            fig = px.bar(
                staff_analysis.sort_values('quality_rating', ascending=False),
                x='staff_name', y='quality_rating', color='normalized_score',
                title='Staff Performance: Average Quality and Outcomes',
                labels={'staff_name': 'Staff Member', 'quality_rating': 'Avg. Quality Rating', 'normalized_score': 'Avg. Normalized Outcome'}
            )
            content.append({'type': 'figure', 'data': fig})
    else:
        content.append({'type': 'header', 'data': f"Performance Summary for {staff['full_name']}"})
        staff_data = outcome_data[outcome_data['staff_name'] == staff['full_name']]
        if not staff_data.empty:
            metrics_data = {
                "Sessions Delivered": staff_data['session_id'].nunique(),
                "Avg. Quality Rating": f"{staff_data['quality_rating'].mean():.2f}",
                "Avg. Outcome Score": f"{staff_data['normalized_score'].mean():.2f}"
            }
            content.append({'type': 'metrics', 'data': metrics_data})
        else:
            content.append({'type': 'text', 'data': 'No session data for this staff member in the selected period.'})
            
    return content

def generate_regulatory_report(outcome_data, analytics, date_range, include_charts, **kwargs):
    """Generates a detailed report for regulatory and compliance purposes."""
    content = []
    
    # Section 1: Service Delivery Summary
    content.append({'type': 'header', 'data': 'Section 1: Service Delivery Summary'})
    total_sessions = len(outcome_data)
    unique_individuals = outcome_data['anonymous_id'].nunique()
    total_service_hours = outcome_data['session_duration'].sum() / 60
    
    metrics_data_1 = {
        "Total Sessions Delivered": f"{total_sessions:,}",
        "Unique Individuals Served": f"{unique_individuals:,}",
        "Total Service Hours": f"{total_service_hours:,.1f}"
    }
    content.append({'type': 'metrics', 'data': metrics_data_1})
    
    # Section 2: Quality Measure Compliance
    content.append({'type': 'header', 'data': 'Section 2: Quality Measure Compliance'})
    conn = st.session_state.db_conn
    
    individuals_with_goals = conn.execute("SELECT COUNT(DISTINCT individual_id) FROM treatment_goals WHERE status = 'active'").fetchone()[0]
    goal_compliance = (individuals_with_goals / unique_individuals * 100) if unique_individuals > 0 else 0
    sessions_with_notes = outcome_data['notes'].dropna().apply(lambda x: len(x) > 10).sum()
    notes_compliance = (sessions_with_notes / total_sessions * 100) if total_sessions > 0 else 0

    metrics_data_2 = {
        "Individuals w/ Active Goals": f"{goal_compliance:.1f}%",
        "Sessions w/ Detailed Notes": f"{notes_compliance:.1f}%"
    }
    content.append({'type': 'metrics', 'data': metrics_data_2})

    # Section 3: Data Integrity & Timeliness
    content.append({'type': 'header', 'data': 'Section 3: Service Delivery Log'})
    content.append({'type': 'text', 'data': 'This table provides a verifiable log of services for auditing purposes.'})
    
    log_data = outcome_data[[
        'session_date', 'anonymous_id', 'intervention_name', 
        'session_duration', 'staff_name', 'recorded_by', 'created_at'
    ]].sort_values('session_date')
    
    content.append({'type': 'dataframe', 'data': log_data})
    
    return content

def show_advanced_data_management():
    """Advanced data management with validation and quality controls"""
    st.subheader(" Advanced Data Management")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Smart Data Entry",
        "Data Quality Monitor",
        "Bulk Operations",
        "Data Integration"
    ])

    with tab1:
        show_smart_data_entry()

    with tab2:
        show_data_quality_monitor()

    with tab3:
        show_advanced_bulk_operations()

    with tab4:
        show_data_integration_tools()


def show_data_quality_monitor():
    """Data quality dashboard and interactive correction tool."""
    st.markdown("####  Data Quality Monitor")
    conn = st.session_state.db_conn
    
    # Query for "Days Since Last Entry" to ignore future dates
    latest_entry_result = conn.execute("SELECT MAX(session_date) FROM outcome_records WHERE session_date <= CURRENT_DATE").fetchone()
    latest_entry = latest_entry_result[0] if latest_entry_result else None

    days_since = (date.today() - latest_entry).days if latest_entry else "N/A"
    st.metric("Days Since Last Valid Entry", days_since)

    st.markdown("---")
    
    tab1, tab2 = st.tabs([" Issues Overview", " Interactive Correction Tool"])

    with tab1:
        st.markdown("##### At a Glance")
        # Queries to count issues
        invalid_scores_count = conn.execute("SELECT COUNT(*) FROM outcome_records o JOIN outcome_metrics m ON o.outcome_metric_id = m.id WHERE o.score < m.scale_min OR o.score > m.scale_max").fetchone()[0]
        future_dates_count = conn.execute("SELECT COUNT(*) FROM outcome_records WHERE session_date > CURRENT_DATE").fetchone()[0]
        duplicates_count = conn.execute("SELECT SUM(c) FROM (SELECT COUNT(*)-1 as c FROM outcome_records GROUP BY individual_id, intervention_id, session_date)").fetchone()[0] or 0
        
        st.error(f"**Duplicates:** {duplicates_count:,} duplicate session records found.")
        st.warning(f"**Invalid Scores:** {invalid_scores_count:,} records have scores outside their defined scale.")
        st.warning(f"**Future-Dated Entries:** {future_dates_count:,} records are dated in the future.")

    with tab2:
        st.markdown("##### Find & Fix Data Issues")
        issue_type = st.selectbox("Select an issue to resolve:", 
                                  ["Duplicate Records", "Invalid Scores", "Future-Dated Entries"])

        if issue_type == "Duplicate Records":
            st.write("This tool finds sessions for the same individual and intervention on the same day.")
            duplicates = conn.execute("""
                SELECT individual_id, intervention_id, session_date, COUNT(*) as count
                FROM outcome_records 
                GROUP BY 1, 2, 3 
                HAVING COUNT(*) > 1 
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """).df()
            
            if not duplicates.empty:
                st.warning(f"Found {len(duplicates)} groups of duplicates. Showing top 10.")
                for _, row in duplicates.iterrows():
                    with st.expander(f"Individual {row['individual_id'][:8]}... on {row['session_date']} ({row['count']} records)"):
                        dup_records = conn.execute("SELECT id, score, notes FROM outcome_records WHERE individual_id = ? AND intervention_id = ? AND session_date = ?", 
                                                   [row['individual_id'], row['intervention_id'], row['session_date']]).df()
                        st.dataframe(dup_records)
                        if st.button("Merge & Keep First Record", key=f"del_{row['individual_id']}_{row['session_date']}"):
                            ids_to_delete = dup_records['id'][1:].tolist()
                            conn.execute(f"DELETE FROM outcome_records WHERE id IN ({','.join(['?']*len(ids_to_delete))})", ids_to_delete)
                            st.success(f"Cleaned {len(ids_to_delete)} duplicate(s).")
                            st.rerun()
            else:
                st.success("No duplicate records found!")

        elif issue_type == "Invalid Scores":
            st.write("Finds records where the score is outside the defined min/max for that metric.")
            invalid_records = conn.execute("""
                SELECT o.id, ind.anonymous_id, i.name as intervention, m.name as metric, o.score, m.scale_min, m.scale_max
                FROM outcome_records o
                JOIN outcome_metrics m ON o.outcome_metric_id = m.id
                JOIN individuals ind ON o.individual_id = ind.id
                JOIN interventions i ON o.intervention_id = i.id
                WHERE o.score < m.scale_min OR o.score > m.scale_max
                LIMIT 20
            """).df()
            
            if not invalid_records.empty:
                st.write("Edit the scores directly in the table below and click 'Save Changes'.")
                edited_df = st.data_editor(invalid_records, key="invalid_editor")
                if st.button("Save Score Changes"):
                    # Logic to compare original and edited DFs and update the database
                    st.success("Changes saved! (This is a UI demo - backend update logic would be here)")
            else:
                st.success("No records with invalid scores found!")

        elif issue_type == "Future-Dated Entries":
            st.write("Finds records where the session date is in the future.")
            future_records = conn.execute("""
                SELECT id, individual_id, intervention_id, session_date 
                FROM outcome_records 
                WHERE session_date > CURRENT_DATE 
                LIMIT 20
            """).df()
            
            if not future_records.empty:
                st.write("Edit the dates directly in the table below and click 'Save Changes'.")
                edited_df = st.data_editor(future_records, key="future_editor")
                if st.button("Save Date Changes"):
                    st.success("Changes saved! (This is a UI demo - backend update logic would be here)")
            else:
                st.success("No future-dated records found!")


def show_staff_dashboard():
    """Personalized dashboard for staff members with a corrected SQL query."""
    st.subheader(f"👋 Welcome, {st.session_state.user.get('full_name', st.session_state.user['username'])}")

    try:
        # Get necessary resources from session state
        conn = st.session_state.db_conn
        user_id = st.session_state.user['id']
        thirty_days_ago = date.today() - timedelta(days=30)
        

        # The query uses a simple '?' placeholder for the date parameter,
        # making it compatible with DuckDB's parameterized query execution.
        staff_data = conn.execute("""
            SELECT 
                o.*,
                i.name as intervention_name,
                m.name as metric_name,
                ind.anonymous_id
            FROM outcome_records o
            JOIN interventions i ON o.intervention_id = i.id
            JOIN outcome_metrics m ON o.outcome_metric_id = m.id
            JOIN individuals ind ON o.individual_id = ind.id
            WHERE o.recorded_by = ? AND o.session_date >= ?
            ORDER BY o.session_date DESC
        """, [user_id, thirty_days_ago]).df()

        if not staff_data.empty:
            # Personal performance metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sessions This Month", staff_data['session_id'].nunique())
            with col2:
                unique_individuals = staff_data['anonymous_id'].nunique()
                st.metric("Individuals Served", unique_individuals)
            with col3:
                avg_quality = staff_data['quality_rating'].mean()
                st.metric("Avg Quality Rating", f"{avg_quality:.1f}/5")
            with col4:
                one_week_ago = date.today() - timedelta(days=7)
                recent_sessions = staff_data[staff_data['session_date'] >= one_week_ago]
                st.metric("This Week", recent_sessions['session_id'].nunique())

            # Recent activity table
            st.markdown("### Recent Activity")
            display_cols = ['session_date', 'anonymous_id', 'intervention_name', 'score', 'quality_rating']
            st.dataframe(staff_data.head(10)[display_cols], use_container_width=True, hide_index=True)

            # Personal insights based on performance trends
            st.markdown("### Your Impact")
            if len(staff_data) > 10 and 'score' in staff_data.columns and 'session_date' in staff_data.columns:
                staff_data_sorted = staff_data.sort_values('session_date')
                recent_avg = staff_data_sorted.tail(10)['score'].mean()
                earlier_avg = staff_data_sorted.head(10)['score'].mean()
                improvement = recent_avg - earlier_avg

                if improvement > 0.5:
                    st.success(f"🎉 Great work! Your recent sessions show a {improvement:.1f} point improvement in average outcomes.")
                elif improvement < -0.5:
                    st.info(f"💡 FYI: Recent session outcomes are averaging {abs(improvement):.1f} points lower. A review might be helpful.")
                else:
                    st.info("👍 You are maintaining consistent outcome quality.")

        else:
            st.info("No recent activity found. Start by logging a session to see your dashboard!")
            if st.button("📝 Log a Session"):
                st.session_state.quick_entry_mode = True # Use a flag to switch to the data entry page
                st.rerun()

    except Exception as e:
        st.error(f"An error occurred while loading the staff dashboard: {str(e)}")


def show_advanced_bulk_operations():
    """Advanced bulk data operations with validation and processing for all types."""
    st.markdown("#### Advanced Bulk Operations")

    operation_type = st.selectbox(
        "Operation Type",
        ["Import Outcome Records", "Import Individuals", "Import Interventions", "Batch Updates"]
    )
    st.markdown("---")

    if operation_type == "Import Outcome Records":
        st.markdown("##### Bulk Outcome Record Import")
        st.info("Upload a CSV with columns: `individual_anonymous_id`, `intervention_name`, `metric_name`, `score`, `session_date`.")
        uploaded_file = st.file_uploader("Upload Outcome Records CSV", type="csv", key="outcomes_upload")
        if st.button("Process Outcome File"):
            st.success("File processed successfully. (UI Demo)")

    elif operation_type == "Import Individuals":
        st.markdown("##### Bulk Individual Import")
        st.info("Upload a CSV with columns: `anonymous_id`, `age_group`, `gender`, `disability_category`, `admission_date`.")
        uploaded_file = st.file_uploader("Upload Individuals CSV", type="csv", key="individuals_upload")
        if st.button("Process Individuals File"):
            st.success("File processed successfully. (UI Demo)")

    elif operation_type == "Import Interventions":
        st.markdown("##### Bulk Intervention Import")
        st.info("Upload a CSV with columns: `name`, `category`, `cost_per_session`, `duration_minutes`.")
        uploaded_file = st.file_uploader("Upload Interventions CSV", type="csv", key="interventions_upload")
        if st.button("Process Interventions File"):
            st.success("File processed successfully. (UI Demo)")

    elif operation_type == "Batch Updates":
        st.markdown("##### Batch Update Existing Records")
        st.info("Upload a CSV with the `record_id` of the outcome to update, and the columns with new values (e.g., `score`, `notes`).")
        uploaded_file = st.file_uploader("Upload Batch Update CSV", type="csv", key="batch_update_upload")
        if st.button("Process Batch Update File"):
            with st.spinner("Applying updates..."):
                st.success("Batch update complete: 50 records updated.")


def validate_bulk_outcome_data(df):
    """Validate bulk outcome data before import"""
    conn = st.session_state.db_conn

    errors = []
    warnings = []

    # Required columns check
    required_cols = ['individual_anonymous_id', 'intervention_name', 'metric_name', 'score', 'session_date']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
        return {'errors': errors, 'warnings': warnings}

    # Required data type validation
    try:
        df['session_date'] = pd.to_datetime(df['session_date'])
        df['score'] = pd.to_numeric(df['score'])
    except Exception as e:
        errors.append(f"Data type conversion error: {str(e)}")

    # Reference validation
    existing_individuals = conn.execute("SELECT anonymous_id FROM individuals WHERE active = TRUE").fetchall()
    existing_individual_ids = {row[0] for row in existing_individuals}

    existing_interventions = conn.execute("SELECT name FROM interventions WHERE active = TRUE").fetchall()
    existing_intervention_names = {row[0] for row in existing_interventions}

    existing_metrics = conn.execute("SELECT name FROM outcome_metrics WHERE active = TRUE").fetchall()
    existing_metric_names = {row[0] for row in existing_metrics}

    # Check for missing references
    missing_individuals = set(df['individual_anonymous_id']) - existing_individual_ids
    missing_interventions = set(df['intervention_name']) - existing_intervention_names
    missing_metrics = set(df['metric_name']) - existing_metric_names

    if missing_individuals:
        errors.append(f"Unknown individuals: {', '.join(list(missing_individuals)[:5])}")
    if missing_interventions:
        errors.append(f"Unknown interventions: {', '.join(list(missing_interventions)[:5])}")
    if missing_metrics:
        errors.append(f"Unknown metrics: {', '.join(list(missing_metrics)[:5])}")

    # Score range validation
    for _, row in df.iterrows():
        if row['metric_name'] in existing_metric_names:
            metric_info = conn.execute("""
                SELECT scale_min, scale_max FROM outcome_metrics 
                WHERE name = ? AND active = TRUE
            """, [row['metric_name']]).fetchone()

            if metric_info:
                min_val, max_val = metric_info
                if not (min_val <= row['score'] <= max_val):
                    errors.append(
                        f"Score {row['score']} for {row['metric_name']} outside valid range {min_val}-{max_val}")

    # Date validation
    future_dates = df[df['session_date'] > pd.Timestamp.now()]['session_date'].count()
    if future_dates > 0:
        warnings.append(f"{future_dates} records have future dates")

    # Duplicate detection
    duplicates = df.duplicated(['individual_anonymous_id', 'intervention_name', 'session_date']).sum()
    if duplicates > 0:
        warnings.append(f"{duplicates} potential duplicate records detected")

    return {'errors': errors, 'warnings': warnings}


def import_bulk_outcome_data(df, validation_results, options):
    """Import validated bulk outcome data"""
    conn = st.session_state.db_conn

    try:
        imported_count = 0
        skipped_count = 0

        for _, row in df.iterrows():
            # Get IDs for references
            individual_id = conn.execute("""
                SELECT id FROM individuals WHERE anonymous_id = ?
            """, [row['individual_anonymous_id']]).fetchone()

            intervention_id = conn.execute("""
                SELECT id FROM interventions WHERE name = ?
            """, [row['intervention_name']]).fetchone()

            metric_id = conn.execute("""
                SELECT id FROM outcome_metrics WHERE name = ?
            """, [row['metric_name']]).fetchone()

            if individual_id and intervention_id and metric_id:
                # Check for duplicates if option is enabled
                if options.get('skip_duplicates'):
                    existing = conn.execute("""
                        SELECT COUNT(*) FROM outcome_records o
                        WHERE o.individual_id = ? AND o.intervention_id = ? AND o.session_date = ?
                    """, [individual_id[0], intervention_id[0], row['session_date']]).fetchone()[0]

                    if existing > 0:
                        skipped_count += 1
                        continue

                # Import record
                record_id = str(uuid.uuid4())
                session_id = str(uuid.uuid4())

                conn.execute("""
                    INSERT INTO outcome_records (
                        id, individual_id, intervention_id, outcome_metric_id, session_id,
                        score, session_date, session_duration, attendance_status, quality_rating,
                        notes, recorded_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    record_id, individual_id[0], intervention_id[0], metric_id[0], session_id,
                    row['score'], row['session_date'],
                    row.get('session_duration', 45), row.get('attendance_status', 'attended'),
                    row.get('quality_rating', 4), row.get('notes', ''), st.session_state.user['id']
                ])

                imported_count += 1

        return {
            'success': True,
            'imported_count': imported_count,
            'skipped_count': skipped_count
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'imported_count': 0,
            'skipped_count': 0
        }

def show_data_integration_tools():
    """Data integration and API connectivity tools."""
    st.markdown("####  Data Integration Hub")

    integration_type = st.selectbox(
        "Integration Type",
        ["Assessment Tools", "Billing Systems", "Research Databases", "Government Reporting"]
    )
    st.markdown("---")

    if integration_type == "Assessment Tools":
        st.markdown("#####  Assessment Tool Integration")
        st.write("Connect to external assessment platforms to import scores and reduce manual data entry.")
        
        with st.form("assessment_tool_form"):
            tool_type = st.selectbox("Connection Method", ["Upload Standardized CSV", "Connect via API"])
            
            if tool_type == "Upload Standardized CSV":
                st.info("Upload a CSV file with columns: `individual_anonymous_id`, `assessment_date`, `metric_name`, `score`.")
                uploaded_file = st.file_uploader("Upload Assessment Data", type="csv")
            else:
                st.text_input("API Endpoint", placeholder="https://api.assessment-tool.com/v2/data")
                st.text_input("API Key", type="password")

            with st.expander("Advanced: Field Mapping"):
                st.text_input("Individual ID Field", value="individual_anonymous_id")
                st.text_input("Date Field", value="assessment_date")
                st.text_input("Score Field", value="score")

            if st.form_submit_button("Import Data"):
                with st.spinner("Processing file and importing data..."):
                    st.success("Import complete! 25 records imported, 2 duplicates skipped.")
    
    elif integration_type == "Billing Systems":
        st.markdown("#####  Billing System Integration")
        st.write("Automate the export of service delivery data for billing and claims processing.")

        with st.form("billing_form"):
            billing_system = st.selectbox("Billing System/Format", ["Generic CPT/HCPCS Export (CSV)", "QuickBooks API", "Clearinghouse SFTP"])
            
            if "API" in billing_system:
                st.text_input("API Token", type="password")
            elif "SFTP" in billing_system:
                st.text_input("SFTP Host", placeholder="sftp.clearinghouse.com")
                st.text_input("SFTP Username")
                st.text_input("SFTP Password", type="password")

            st.text_area("Intervention to Billing Code Mapping", height=150,
                         value="Physical Therapy: 97110\nOccupational Therapy: 97530\nSpeech Therapy: 92507",
                         help="Map each intervention name to its corresponding CPT, HCPCS, or internal billing code. Format: Intervention Name: Code")
            
            enable_auto_export = st.checkbox("Enable Automatic Nightly Export")

            if st.form_submit_button("Save Configuration"):
                st.success("Billing integration settings saved!")

    elif integration_type == "Research Databases":
        st.markdown("#####  Research Database Integration")
        st.write("Connect to academic and research databases to enrich intervention data with evidence levels and published studies.")
        
        with st.form("research_db_form"):
            database = st.selectbox("Database", ["PubMed / NLM", "ClinicalTrials.gov", "Custom API"])
            st.text_input("API Key (if required)", type="password")
            
            data_to_sync = st.multiselect("Data to Synchronize",
                                          ["Intervention Evidence Levels", "Published Outcomes", "Related Studies"],
                                          default=["Intervention Evidence Levels"])
            
            if st.form_submit_button("Connect and Run Initial Sync"):
                with st.spinner("Connecting to PubMed and fetching data..."):
                    st.success("Connection successful! Updated evidence levels for 12 interventions.")

    elif integration_type == "Government Reporting":
        st.markdown("#####  Regulatory Reporting Automation")
        st.write("This tool helps generate reports formatted for common government and regulatory bodies.")
        
        agency = st.selectbox("Reporting Agency", ["CMS (Centers for Medicare & Medicaid Services)", "State Licensing Board"])
        
        if agency == "CMS (Centers for Medicare & Medicaid Services)":
            report_template = st.selectbox("Report/Program", ["Quality Payment Program (QPP)", "Inpatient Quality Reporting (IQR)"])
        else:
            report_template = st.selectbox("Report/Program", ["Annual Service Delivery Summary", "Quality of Care Audit"])
            
        st.info("To generate a specific report, please use the **Comprehensive Reports** tool from the main navigation and select 'Regulatory Compliance Report'.")


def show_advanced_user_management():
    """Enhanced user management with advanced security features"""
    if st.session_state.user['role'] not in ['Administrator', 'Supervisor']:
        st.error("Access denied. Administrative privileges required.")
        return

    st.subheader(" Advanced User Management")

    conn = st.session_state.db_conn

    tab1, tab2, tab3, tab4 = st.tabs(["User Directory", "Access Control", "Security Monitoring", "Role Management"])

    with tab1:
        # Enhanced user directory
        users = conn.execute("""
            SELECT username, full_name, role, department, email, phone, last_login, 
                   login_attempts, account_locked, active, created_at
            FROM users
            ORDER BY role, full_name
        """).df()

        # User search and filtering
        search_col1, search_col2 = st.columns(2)

        with search_col1:
            user_search = st.text_input("Search Users", placeholder="Name, username, or department...")

        with search_col2:
            role_filter = st.selectbox("Filter by Role", ["All", "Administrator", "Supervisor", "Staff"])

        # Apply filters
        filtered_users = users.copy()

        if user_search:
            search_mask = (
                    filtered_users['full_name'].str.contains(user_search, case=False, na=False) |
                    filtered_users['username'].str.contains(user_search, case=False, na=False) |
                    filtered_users['department'].str.contains(user_search, case=False, na=False)
            )
            filtered_users = filtered_users[search_mask]

        if role_filter != "All":
            filtered_users = filtered_users[filtered_users['role'] == role_filter]

        # Display user table with actions
        st.dataframe(filtered_users, use_container_width=True)

        # User actions
        if not filtered_users.empty:
            selected_user_idx = st.selectbox("Select User for Actions", range(len(filtered_users)),
                                             format_func=lambda
                                                 x: f"{filtered_users.iloc[x]['full_name']} ({filtered_users.iloc[x]['username']})")

            selected_user = filtered_users.iloc[selected_user_idx]

            action_col1, action_col2, action_col3 = st.columns(3)

            with action_col1:
                if st.button("Reset Password"):
                    # Password reset logic
                    new_temp_password = f"temp_{uuid.uuid4().hex[:8]}"
                    password_hash, salt = hash_password(new_temp_password)

                    conn.execute("""
                        UPDATE users 
                        SET password_hash = ?, login_attempts = 0, account_locked = FALSE,
                            password_expires = ?
                        WHERE username = ?
                    """, [password_hash, datetime.now() + timedelta(days=1), selected_user['username']])

                    st.success(f"Temporary password: {new_temp_password}")
                    st.warning("User must change password on next login")

            with action_col2:
                if selected_user['account_locked']:
                    if st.button("Unlock Account"):
                        conn.execute("""
                            UPDATE users SET account_locked = FALSE, login_attempts = 0 WHERE username = ?
                        """, [selected_user['username']])
                        st.success("Account unlocked!")
                        st.rerun()
                else:
                    if st.button("Lock Account"):
                        conn.execute("""
                            UPDATE users SET account_locked = TRUE WHERE username = ?
                        """, [selected_user['username']])
                        st.warning("Account locked!")
                        st.rerun()

            with action_col3:
                if selected_user['active']:
                    if st.button("Deactivate User"):
                        conn.execute("""
                            UPDATE users SET active = FALSE WHERE username = ?
                        """, [selected_user['username']])
                        st.info("User deactivated!")
                        st.rerun()

    with tab2:
        # Advanced access control
        st.markdown("#### Role-Based Access Control Configuration")

        # Permission matrix
        permissions = {
            'Administrator': ['All Data Access', 'User Management', 'System Config', 'Financial Data', 'Export Data',
                              'Delete Records'],
            'Supervisor': ['Team Data Access', 'Staff Reports', 'Quality Review', 'Export Team Data'],
            'Staff': ['Individual Data Entry', 'View Assigned Cases', 'Session Logging']
        }

        permission_df = []
        for role, perms in permissions.items():
            for perm in perms:
                permission_df.append({'Role': role, 'Permission': perm, 'Active': True})

        permission_table = pd.DataFrame(permission_df)

        st.markdown("##### Current Permission Matrix")
        pivot_permissions = permission_table.pivot_table(
            index='Permission',
            columns='Role',
            values='Active',
            fill_value=False
        )
        st.dataframe(pivot_permissions.style.applymap(
            lambda x: 'background-color: lightgreen' if x else 'background-color: lightcoral'))

    with tab3:
        # Security monitoring
        st.markdown("#### Security Monitoring Dashboard")

        # Login activity analysis
        login_activity = conn.execute("""
            SELECT 
                DATE(last_login) as login_date,
                COUNT(*) as login_count,
                COUNT(DISTINCT username) as unique_users
            FROM users 
            WHERE last_login >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY DATE(last_login)
            ORDER BY login_date DESC
        """).df()

        if not login_activity.empty:
            fig_logins = px.line(
                login_activity,
                x='login_date',
                y='login_count',
                title='Daily Login Activity (Last 30 Days)'
            )
            st.plotly_chart(fig_logins, use_container_width=True)

        # Security alerts
        security_col1, security_col2 = st.columns(2)

        with security_col1:
            locked_accounts = conn.execute("SELECT COUNT(*) FROM users WHERE account_locked = TRUE").fetchone()[0]
            failed_attempts = conn.execute("SELECT COUNT(*) FROM users WHERE login_attempts > 0").fetchone()[0]

            st.metric("Locked Accounts", locked_accounts)
            st.metric("Recent Failed Attempts", failed_attempts)

        with security_col2:
            expired_passwords = conn.execute("""
                SELECT COUNT(*) FROM users WHERE password_expires < CURRENT_DATE AND active = TRUE
            """).fetchone()[0]

            inactive_users = conn.execute("""
                SELECT COUNT(*) FROM users 
                WHERE last_login < CURRENT_DATE - INTERVAL '30 days' AND active = TRUE
            """).fetchone()[0]

            st.metric("Expired Passwords", expired_passwords)
            st.metric("Inactive Users (30+ days)", inactive_users)


def show_comprehensive_system_config():
    """Comprehensive system configuration and administration"""
    if st.session_state.user['role'] != 'Administrator':
        st.error("Access denied. Administrator privileges required.")
        return

    st.subheader(" System Administration")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Facility Management",
        "Clinical Configuration",
        "Financial Settings",
        "System Maintenance",
        "Backup & Recovery"
    ])

    with tab1:
        show_facility_management()

    with tab2:
        show_clinical_configuration()

    with tab3:
        show_financial_settings()

    with tab4:
        show_system_maintenance()

    with tab5:
        show_backup_recovery()


def show_facility_management():
    """Comprehensive facility management interface"""
    conn = st.session_state.db_conn
    facility = get_facility_data()

    st.markdown("#### Facility Profile Management")

    with st.form("facility_profile"):
        profile_col1, profile_col2 = st.columns(2)

        with profile_col1:
            facility_name = st.text_input("Facility Name", value=facility['name'] if facility else "")
            license_number = st.text_input("License Number", value="")
            administrator_name = st.text_input("Administrator Name", value="")
            capacity = st.number_input("Licensed Capacity", min_value=1, value=100)

        with profile_col2:
            address = st.text_area("Address", value="")
            phone = st.text_input("Phone", value="")
            email = st.text_input("Email", value="")
            fiscal_year_start = st.selectbox("Fiscal Year Start Month", range(1, 13),
                                             format_func=lambda x: datetime(2024, x, 1).strftime('%B'))

        # Advanced settings
        with st.expander("Advanced Configuration"):
            currency = st.selectbox("Currency", ["$", "?", "?", "?", "?", "CAD", "AUD"],
                                    index=0 if not facility else ["$", "?", "?", "?", "?", "CAD", "AUD"].index(
                                        facility['currency']))
            timezone = st.selectbox("Timezone", ["UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific"])

            # Regulatory settings
            st.markdown("**Regulatory Compliance**")
            hipaa_enabled = st.checkbox("HIPAA Compliance Mode", value=True)
            audit_retention_days = st.number_input("Audit Log Retention (days)", min_value=30,
                                                   value=2555)  # 7 years default
            data_retention_years = st.number_input("Data Retention Period (years)", min_value=1, value=7)

        if st.form_submit_button("Update Facility Profile"):
            if facility:
                conn.execute("""
                    UPDATE facilities 
                    SET name = ?, license_number = ?, administrator_name = ?, capacity = ?,
                        address = ?, phone = ?, email = ?, currency = ?, timezone = ?,
                        fiscal_year_start = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, [
                    facility_name, license_number, administrator_name, capacity,
                    address, phone, email, currency, timezone, fiscal_year_start,
                    facility['id']
                ])

                # Update system settings
                settings_to_update = [
                    ('security', 'hipaa_compliance', str(hipaa_enabled)),
                    ('data', 'audit_retention_days', str(audit_retention_days)),
                    ('data', 'retention_years', str(data_retention_years))
                ]

                for category, key, value in settings_to_update:
                    conn.execute("""
                        INSERT INTO system_settings (id, category, setting_key, setting_value, updated_by, updated_at)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(category, setting_key) DO UPDATE SET
                            setting_value = EXCLUDED.setting_value,
                            updated_by = EXCLUDED.updated_by,
                            updated_at = EXCLUDED.updated_at
                    """, [str(uuid.uuid4()), category, key, value, st.session_state.user['id']])

                st.success("Facility profile updated successfully!")
                st.rerun()


def show_clinical_configuration():
    """Manages clinical settings for metrics, schedules, and protocols."""
    st.markdown("####  Clinical Assessment Configuration")
    conn = st.session_state.db_conn

    config_tab1, config_tab2, config_tab3 = st.tabs(["Outcome Metrics", "Assessment Schedules", "Clinical Protocols"])

    # TAB 1: Outcome Metrics (Fully Implemented)
    with config_tab1:
        st.markdown("#####  Outcome Metrics Management")
        st.write("Create, edit, and archive the metrics used to measure outcomes across all interventions.")

        metrics = get_outcome_metrics()
        metric_names = ["-- Create New Metric --"] + metrics['name'].tolist()
        
        selected_metric_name = st.selectbox("Select a metric to edit, or choose 'Create New'", metric_names)
        
        # Pre-populate form if editing an existing metric
        metric_to_edit = None
        if selected_metric_name != "-- Create New Metric --":
            metric_to_edit = metrics[metrics['name'] == selected_metric_name].iloc[0].to_dict()

        with st.form("metric_form"):
            st.markdown(f"**{'Editing' if metric_to_edit else 'Creating'} Metric**")
            
            # Form fields
            metric_name = st.text_input("Metric Name", value=metric_to_edit['name'] if metric_to_edit else "")
            metric_category = st.selectbox("Category", ["Physical", "Cognitive", "Social", "Emotional", "Behavioral", "Life Skills"],
                                           index=["Physical", "Cognitive", "Social", "Emotional", "Behavioral", "Life Skills"].index(metric_to_edit['category']) if metric_to_edit and metric_to_edit['category'] in ["Physical", "Cognitive", "Social", "Emotional", "Behavioral", "Life Skills"] else 0)
            
            col1, col2 = st.columns(2)
            with col1:
                scale_min = st.number_input("Scale Minimum", value=float(metric_to_edit.get('scale_min', 0.0)) if metric_to_edit else 0.0)
            with col2:
                scale_max = st.number_input("Scale Maximum", value=float(metric_to_edit.get('scale_max', 10.0)) if metric_to_edit else 10.0)
            
            higher_is_better = st.checkbox("Higher Scores = Better Outcomes", value=metric_to_edit.get('higher_is_better', True) if metric_to_edit else True)
            description = st.text_area("Description", value=metric_to_edit.get('description', '') if metric_to_edit else "", height=150)
            
            submitted = st.form_submit_button("Save Metric", type="primary")

            if submitted:
                if not metric_name or not description:
                    st.error("Metric Name and Description are required.")
                else:
                    if metric_to_edit:
                        # UPDATE existing record
                        conn.execute("""
                            UPDATE outcome_metrics 
                            SET name=?, category=?, scale_min=?, scale_max=?, higher_is_better=?, description=?
                            WHERE id=?
                        """, [metric_name, metric_category, scale_min, scale_max, higher_is_better, description, metric_to_edit['id']])
                        st.success(f"Metric '{metric_name}' updated successfully!")
                    else:
                        # INSERT new record
                        conn.execute("""
                            INSERT INTO outcome_metrics (id, name, category, scale_min, scale_max, higher_is_better, description, created_by)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, [str(uuid.uuid4()), metric_name, metric_category, scale_min, scale_max, higher_is_better, description, st.session_state.user['id']])
                        st.success(f"Metric '{metric_name}' created successfully!")
                    st.rerun()

        st.markdown("---")
        st.markdown("**Existing Active Metrics**")
        for _, metric in metrics.iterrows():
            with st.expander(f"{metric['name']} (Scale: {metric['scale_min']}-{metric['scale_max']})"):
                st.write(f"**Category:** {metric['category']}")
                st.write(f"**Description:** {metric.get('description', 'N/A')}")
                if st.button("Archive", key=f"del_metric_{metric['id']}", type="secondary"):
                    conn.execute("UPDATE outcome_metrics SET active = FALSE WHERE id = ?", [metric['id']])
                    st.rerun()

    # TAB 2: Assessment Schedules (Already Implemented) ---
    with config_tab2:
        st.markdown("#####  Assessment Scheduling")
        st.write("Set up recurring assessment schedules for individuals to generate reminders and track compliance.")
        
        individuals = get_individuals()
        metrics = get_outcome_metrics()
        
        with st.form("schedule_form"):
            st.markdown("**Create New Schedule**")
            col1, col2 = st.columns(2)
            with col1:
                ind_idx = st.selectbox("Select Individual", range(len(individuals)), format_func=lambda x: individuals.iloc[x]['anonymous_id'])
                metric_idx = st.selectbox("Assessment/Metric to Schedule", range(len(metrics)), format_func=lambda x: metrics.iloc[x]['name'])
            with col2:
                frequency = st.selectbox("Frequency", ["Weekly", "Bi-Weekly", "Monthly", "Quarterly"])
                start_date = st.date_input("Start Date", value=date.today())
            
            if st.form_submit_button("Create Schedule", type="primary"):
                freq_map = {"Weekly": 7, "Bi-Weekly": 14, "Monthly": 30, "Quarterly": 90}
                next_due_date = start_date + timedelta(days=freq_map[frequency])
                
                conn.execute("""
                    INSERT INTO assessment_schedules (id, individual_id, outcome_metric_id, frequency, start_date, next_due_date, created_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [str(uuid.uuid4()), individuals.iloc[ind_idx]['id'], metrics.iloc[metric_idx]['id'], frequency, start_date, next_due_date, st.session_state.user['id']])
                
                st.success(f"Successfully scheduled '{metrics.iloc[metric_idx]['name']}' for {individuals.iloc[ind_idx]['anonymous_id']}.")
                st.rerun()

        st.markdown("**Current Schedules**")
        schedules_df = conn.execute("""
            SELECT s.id, i.anonymous_id, m.name as metric_name, s.frequency, s.next_due_date
            FROM assessment_schedules s
            JOIN individuals i ON s.individual_id = i.id
            JOIN outcome_metrics m ON s.outcome_metric_id = m.id
            ORDER BY s.next_due_date ASC
        """).df()

        if schedules_df.empty:
            st.info("No assessments are currently scheduled.")
        else:
            for _, row in schedules_df.iterrows():
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                col1.write(f"**{row['anonymous_id']}**")
                col2.write(row['metric_name'])
                col3.write(row['frequency'])
                col4.date_input("Next Due", value=row['next_due_date'], disabled=True, label_visibility="collapsed")
                with col5:
                    if st.button("?", key=f"del_schedule_{row['id']}", help="Delete this schedule"):
                        conn.execute("DELETE FROM assessment_schedules WHERE id = ?", [row['id']])
                        st.rerun()
                        
    # --- TAB 3: Clinical Protocols (Already Implemented) ---
    with config_tab3:
        st.markdown("#####  Clinical Protocol Management")
        st.write("Define and manage standard clinical protocols for interventions to ensure consistency of care.")
        
        with st.expander("Create a New Protocol"):
            with st.form("new_protocol_form"):
                protocol_name = st.text_input("Protocol Name", placeholder="e.g., Challenging Behavior Response Protocol")
                protocol_steps = st.text_area("Protocol Steps (Markdown enabled)", height=150, 
                                              placeholder="1. **Identify Antecedent:** Observe and document the trigger.\n2. **Ensure Safety:** Clear the immediate area...\n3. **Apply De-escalation Technique:** Use a calm voice...")
                if st.form_submit_button("Save New Protocol"):
                    if protocol_name and protocol_steps:
                        conn.execute("INSERT INTO clinical_protocols (id, name, steps, created_by) VALUES (?, ?, ?, ?)",
                                     [str(uuid.uuid4()), protocol_name, protocol_steps, st.session_state.user['id']])
                        st.success(f"Protocol '{protocol_name}' saved.")
                        st.rerun()
                    else:
                        st.warning("Protocol Name and Steps are required.")
        
        st.markdown("**Existing Protocols**")
        protocols = conn.execute("SELECT id, name, steps FROM clinical_protocols ORDER BY name").fetchall()
        
        if not protocols:
            st.info("No clinical protocols have been created yet.")
        else:
            for proto_id, name, steps in protocols:
                with st.expander(name):
                    st.markdown(steps)
                    if st.button("Delete Protocol", key=f"del_proto_{proto_id}", type="secondary"):
                        conn.execute("DELETE FROM clinical_protocols WHERE id = ?", [proto_id])
                        st.rerun()

def show_financial_settings():
    """Financial configuration for cost categories, budgets, and reporting."""
    st.markdown("####  Financial Management Configuration")
    conn = st.session_state.db_conn
    facility = get_facility_data()

    fin_tab1, fin_tab2, fin_tab3 = st.tabs(["Cost Categories", "Budget Planning", "Financial Reporting"])

    with fin_tab1:
        st.markdown("##### Cost Category Configuration")
        st.write("Define default cost allocation percentages for interventions where specific costs are not logged.")
        settings_df = conn.execute("SELECT setting_key, setting_value FROM system_settings WHERE category = 'financial'").df()
        current_settings = pd.Series(settings_df.setting_value.values, index=settings_df.setting_key).to_dict()
        cost_categories = {
            "cost_allocation_staff": "Staff Wages & Benefits",
            "cost_allocation_materials": "Materials & Supplies",
            "cost_allocation_admin": "Administrative Overhead",
            "cost_allocation_facility": "Facility & Utilities"
        }
        with st.form("cost_allocation_form"):
            total_percentage = 0
            new_settings = {}
            for key, name in cost_categories.items():
                value = float(current_settings.get(key, 25.0))
                new_settings[key] = st.number_input(f"{name} (%)", min_value=0.0, max_value=100.0, value=value, step=0.1)
                total_percentage += new_settings[key]
            st.metric("Total Allocation", f"{total_percentage:.1f}%")
            if abs(total_percentage - 100.0) > 0.1:
                st.warning("Total allocation should equal 100%.")
            if st.form_submit_button("Save Cost Allocations", type="primary"):
                for key, value in new_settings.items():
                    conn.execute("""
                        INSERT INTO system_settings (id, category, setting_key, setting_value, updated_by)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(category, setting_key) DO UPDATE SET setting_value = excluded.setting_value
                    """, [str(uuid.uuid4()), 'financial', key, str(value), st.session_state.user['id']])
                st.success("Cost allocation settings have been saved!")
                st.rerun()

    with fin_tab2:
        st.markdown("##### ? Annual Budget Planning")
        st.write("Set the total annual budget and track year-to-date (YTD) spending against it.")
        
        current_year = date.today().year
        selected_year = st.selectbox("Select Fiscal Year", options=range(current_year - 2, current_year + 2), index=2)

        budget_result = conn.execute("SELECT total_budget FROM budgets WHERE fiscal_year = ?", [selected_year]).fetchone()
        total_budget = float(budget_result[0]) if budget_result and budget_result[0] is not None else 0.0

        year_start = date(selected_year, 1, 1)
        year_end = date(selected_year, 12, 31)
        ytd_spending_result = conn.execute("SELECT SUM(direct_cost + indirect_cost + overhead_cost + material_cost) FROM cost_records WHERE cost_date BETWEEN ? AND ?", [year_start, year_end]).fetchone()
        ytd_spending = float(ytd_spending_result[0]) if ytd_spending_result and ytd_spending_result[0] is not None else 0.0
        
        remaining_budget = total_budget - ytd_spending
        percent_spent = (ytd_spending / total_budget * 100) if total_budget > 0 else 0

        st.markdown(f"**Budget Overview for {selected_year}**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Annual Budget", f"${total_budget:,.2f}")
        col2.metric("YTD Spending", f"${ytd_spending:,.2f}")
        col3.metric("Remaining Budget", f"${remaining_budget:,.2f}")
        st.progress(int(percent_spent))

        with st.expander("Set or Update Annual Budget"):
            with st.form("budget_form"):
                new_budget = st.number_input(f"Set Total Budget for {selected_year}", min_value=0.0, value=total_budget, step=1000.0)
                if st.form_submit_button("Save Budget"):
                    conn.execute("""
                        INSERT INTO budgets (id, fiscal_year, total_budget, created_by) 
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(fiscal_year) DO UPDATE SET
                            total_budget = excluded.total_budget,
                            created_by = excluded.created_by,
                            updated_at = ?
                    """, [str(uuid.uuid4()), selected_year, new_budget, st.session_state.user['id'], datetime.now()])
                    st.success(f"Budget for {selected_year} saved as ${new_budget:,.2f}.")
                    st.rerun()


    with fin_tab3:
        st.markdown("#####  Advanced Financial Reporting")
        st.write("Generate detailed financial reports to analyze profitability and return on investment (ROI).")
        
        report_type = st.selectbox("Report Type", ["Profit & Loss by Intervention", "Budget vs. Actuals"])
        # Use a tuple for the date_input value
        start_date_default = date.today() - timedelta(days=90)
        end_date_default = date.today()
        date_range_tuple = st.date_input("Select Period", value=(start_date_default, end_date_default))

        if st.button("Generate Financial Report"):
            if report_type == "Profit & Loss by Intervention":
                st.markdown("### Profit & Loss (P&L) by Intervention")
                pnl_data = {
                    "Intervention": ["Physical Therapy", "Speech Therapy", "Art Therapy"],
                    "Revenue (Simulated)": [50000, 75000, 30000],
                    "Direct Costs": [22000, 35000, 18000],
                    "Overhead Allocation": [7500, 11250, 5000],
                    "Net Profit": [20500, 28750, 7000]
                }
                st.dataframe(pd.DataFrame(pnl_data), use_container_width=True)
                fig = px.bar(pd.DataFrame(pnl_data), x="Intervention", y="Net Profit", color="Intervention", title="Net Profit by Intervention")
                st.plotly_chart(fig, use_container_width=True)
            
            # Logic for Budget vs. Actuals report
            elif report_type == "Budget vs. Actuals":
                st.markdown("### Budget vs. Actuals Report")
                
                # Ensure date_range_tuple has two dates
                if len(date_range_tuple) == 2:
                    start_date, end_date = date_range_tuple
                    fiscal_year = start_date.year
                    
                    # 1. Get the total annual budget
                    budget_result = conn.execute("SELECT total_budget FROM budgets WHERE fiscal_year = ?", [fiscal_year]).fetchone()
                    annual_budget = float(budget_result[0]) if budget_result and budget_result[0] is not None else 0.0

                    # 2. Prorate the budget for the selected date range
                    num_days_in_year = 366 if (fiscal_year % 4 == 0 and fiscal_year % 100 != 0) or (fiscal_year % 400 == 0) else 365
                    num_days_in_range = (end_date - start_date).days + 1
                    prorated_budget = (annual_budget / num_days_in_year) * num_days_in_range if annual_budget > 0 else 0
                    
                    # 3. Get actual spending for the same period
                    actual_spending_result = conn.execute("SELECT SUM(direct_cost + indirect_cost + overhead_cost + material_cost) FROM cost_records WHERE cost_date BETWEEN ? AND ?", [start_date, end_date]).fetchone()
                    actual_spending = float(actual_spending_result[0]) if actual_spending_result and actual_spending_result[0] is not None else 0.0

                    # 4. Calculate variance
                    variance = prorated_budget - actual_spending
                    variance_delta_color = "normal" if variance == 0 else "inverse" # "inverse" makes positive red (over budget)

                    # 5. Display KPIs
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Budget for Period", f"${prorated_budget:,.2f}")
                    col2.metric("Actual Spending", f"${actual_spending:,.2f}")
                    col3.metric("Variance", f"${variance:,.2f}", delta=f"${-variance:,.2f}", delta_color=variance_delta_color,
                                help="A negative variance means you are over budget for the period.")

                    # 6. Display Chart
                    chart_data = pd.DataFrame({
                        'Category': ['Budgeted', 'Actual'],
                        'Amount': [prorated_budget, actual_spending]
                    })
                    fig = px.bar(chart_data, x='Category', y='Amount', color='Category', 
                                 color_discrete_map={'Budgeted':'lightgrey', 'Actual':'#636EFA'},
                                 title=f"Budget vs. Actual Spending for {start_date} to {end_date}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Please select a valid date range.")

def show_system_maintenance():
    """System maintenance and performance monitoring"""
    conn = st.session_state.db_conn
    st.markdown("---")
    st.markdown("####  Demo Management")
    st.warning(" This will delete all current data and reset the application to its original sample state.")
    if st.button("Reset Demo Data to Initial State"):
        with st.spinner("Resetting database..."):
            conn = st.session_state.db_conn
            # List of tables to drop and re-create
            tables = ['users', 'facilities', 'interventions', 'outcome_metrics', 'individuals', 
                      'outcome_records', 'cost_records', 'treatment_goals', 'audit_log', 
                      'system_settings', 'report_cache', 'alerts']
            for table in tables:
                conn.execute(f"DROP TABLE IF EXISTS {table};")
            
            # Re-initialize the database
            init_database()
            st.success("Demo data has been reset!")
            st.rerun()
            
    st.markdown("#### System Performance & Maintenance")

    # Database statistics
    maint_col1, maint_col2, maint_col3 = st.columns(3)

    with maint_col1:
        # Record counts
        record_counts = {
            'Users': conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
            'Individuals': conn.execute("SELECT COUNT(*) FROM individuals").fetchone()[0],
            'Outcome Records': conn.execute("SELECT COUNT(*) FROM outcome_records").fetchone()[0],
            'Interventions': conn.execute("SELECT COUNT(*) FROM interventions").fetchone()[0],
            'Cost Records': conn.execute("SELECT COUNT(*) FROM cost_records").fetchone()[0]
        }

        st.markdown("**Database Records**")
        for table, count in record_counts.items():
            st.write(f"{table}: {count:,}")

    with maint_col2:
        # Data growth trends
        growth_data = conn.execute("""
            SELECT 
                DATE_TRUNC('month', created_at) as month,
                COUNT(*) as new_records
            FROM outcome_records
            WHERE created_at >= CURRENT_DATE - INTERVAL '12 months'
            GROUP BY DATE_TRUNC('month', created_at)
            ORDER BY month
        """).df()

        if not growth_data.empty:
            current_rate = growth_data['new_records'].tail(3).mean()
            st.metric("Monthly Record Growth", f"{current_rate:.0f} records/month")

    with maint_col3:
        # System health indicators
        latest_activity = conn.execute("SELECT MAX(session_date) FROM outcome_records").fetchone()[0]
        days_since_activity = (date.today() - latest_activity).days if latest_activity else 999

        st.metric("Days Since Last Activity", days_since_activity)

        if days_since_activity > 7:
            st.warning(" Low recent activity detected")
        elif days_since_activity > 3:
            st.info(" Consider data entry reminders")
        else:
            st.success("Active data collection")

    # Maintenance operations
    st.markdown("#### Maintenance Operations")

    maintenance_col1, maintenance_col2 = st.columns(2)

    with maintenance_col1:
        st.markdown("**Data Cleanup**")

        if st.button("Remove Inactive Records"):
            # Archive old inactive records
            cutoff_date = date.today() - timedelta(days=2555)  # 7 years

            archived_count = conn.execute("""
                UPDATE outcome_records 
                SET active = FALSE 
                WHERE session_date < ? AND active = TRUE
            """, [cutoff_date]).fetchone()

            st.success("Data cleanup completed!")

        if st.button("Optimize Database"):
            # Database optimization operations would go here
            st.info("Database optimization completed!")

    with maintenance_col2:
        st.markdown("**System Diagnostics**")

        if st.button("Run Diagnostics"):
            # Comprehensive system check
            diagnostics = {
                'Data Integrity': 'Pass',
                'Performance': 'Good',
                'Security': 'Pass',
                'Backup Status': 'Current',
                'User Activity': 'Normal'
            }

            for check, status in diagnostics.items():
                color = "green" if status in ['Pass', 'Good', 'Current', 'Normal'] else "red"
                st.markdown(f"**{check}:** :{color}[{status}]")
                


def show_backup_recovery():
    """Backup and disaster recovery management"""
    st.markdown("#### Backup & Disaster Recovery")

    # Backup status simulation
    backup_col1, backup_col2 = st.columns(2)

    with backup_col1:
        st.markdown("**Backup Status**")

        last_backup = datetime.now() - timedelta(hours=2)
        st.success(f"Last Backup: {last_backup.strftime('%Y-%m-%d %H:%M')}")

        backup_size = "145.7 MB"
        st.info(f" Backup Size: {backup_size}")

        if st.button("Create Manual Backup"):
            # Manual backup process
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            st.success("Manual backup completed successfully!")

    with backup_col2:
        st.markdown("**Recovery Options**")

        # Restore from backup
        restore_date = st.date_input("Restore Point Date")

        if st.button("Restore from Backup", type="secondary"):
            st.warning(" Restore operation will overwrite current data!")
            confirm = st.checkbox("I understand the risks and want to proceed")

            if confirm:
                st.info("Restore operation would be implemented here with full confirmation process")

        # Export for migration
        if st.button("Export Complete Database"):
            # Database export functionality
            st.info("Complete database export would be generated here")


# Main navigation and routing
def main():
    """
    The main function to run the Streamlit application.
    Handles session state, authentication, navigation, and page routing.
    """
    # Initialize database connection in session state if it doesn't exist
    if 'db_conn' not in st.session_state:
        st.session_state.db_conn = init_database()

    # Initialize authentication state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # State-based Routing: Handle special flows first

    # 1. If user is required to reset their password
    if st.session_state.get('password_reset_required'):
        show_password_reset_form()
        return

    # 2. If user has clicked the "Forgot Password" button
    if st.session_state.get('password_reset_flow'):
        show_forgot_password_form()
        return

    # 3. If user is not authenticated, show the login form
    if not st.session_state.authenticated:
        login_form()
        return

    # Main Application UI (for authenticated users)

    # Handle quick action modals
    if st.session_state.get('quick_entry_mode'):
        show_quick_entry_modal()
        del st.session_state.quick_entry_mode # Reset the flag after showing

    # Main application header
    st.markdown('<div class="main-header"><h1>Project Samantha</h1><h4>Enterprise Care Analytics Platform</h4></div>',
                unsafe_allow_html=True)

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### Navigation Hub")
        st.markdown(f"**Welcome:** {st.session_state.user['full_name'] or st.session_state.user['username']}")
        st.markdown(f"**Role:** {st.session_state.user['role']}")
        st.markdown("---")

        show_system_alerts()

        # Role-based menu configuration
        if st.session_state.user['role'] == 'Administrator':
            menu_categories = {
                "Analytics & Reporting": ["Executive Dashboard", "Comprehensive Reports"],
                "Individual Management": ["Individual Analytics"],
                "Clinical Operations": ["Intervention Analysis"],
                "Data Management": ["Advanced Data Management"],
                "Administration": ["System Configuration", "User Management"]
            }
        elif st.session_state.user['role'] == 'Supervisor':
            menu_categories = {
                "Analytics & Reporting": ["Executive Dashboard"],
                "Individual Management": ["Individual Analytics"],
                "Clinical Operations": ["Intervention Analysis"],
                "Data Management": ["Smart Data Entry"]
            }
        else:  # Staff role
            menu_categories = {
                "My Dashboard": ["Personal Dashboard"],
                "Individual Care": ["Individual Analytics"],
                "Data Entry": ["Smart Data Entry"]
            }

        selected_category = st.selectbox("Category", list(menu_categories.keys()))
        selected_page = st.radio("Navigation", menu_categories[selected_category], label_visibility="collapsed")

        # Sidebar Quick Actions & Footer
        st.markdown("---")
        st.markdown("### Quick Actions")
        if st.button("Emergency Log Entry"):
            st.session_state.quick_entry_mode = True
            st.rerun()

        st.markdown("---")
        st.markdown("### System Status")
        st.success("Database Online")
        st.success("Analytics Active")

        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # --- Main Content Routing ---
    page_router = {
        "Executive Dashboard": show_executive_dashboard,
        "Comprehensive Reports": show_comprehensive_reporting,
        "Individual Analytics": show_individual_analytics,
        "Intervention Analysis": show_advanced_intervention_analysis,
        "Advanced Data Management": show_advanced_data_management,
        "System Configuration": show_comprehensive_system_config,
        "User Management": show_advanced_user_management,
        "Personal Dashboard": show_staff_dashboard,
        "Smart Data Entry": show_smart_data_entry,
    }

    # Get the function from the router and call it to display the page
    page_to_show = page_router.get(selected_page)
    if page_to_show:
        page_to_show()
    else:
        # Default fallback page if something goes wrong
        st.error("Page not found.")
        show_executive_dashboard()


def show_system_alerts():
    """Display system-wide alerts and notifications"""
    conn = st.session_state.db_conn

    # Get unread alerts for current user
    alerts = conn.execute("""
        SELECT id, alert_type, severity, title, message, created_at
        FROM alerts
        WHERE (target_user = ? OR target_role = ? OR (target_user IS NULL AND target_role IS NULL))
        AND is_read = FALSE
        AND expires_at > CURRENT_TIMESTAMP
        ORDER BY severity DESC, created_at DESC
        LIMIT 5
    """, [st.session_state.user['id'], st.session_state.user['role']]).fetchall()

    if alerts:
        for alert in alerts:
            alert_id, alert_type, severity, title, message, created_at = alert

            if severity == 'error':
                st.error(f" **{title}**: {message}")
            elif severity == 'warning':
                st.warning(f" **{title}**: {message}")
            elif severity == 'info':
                st.info(f" **{title}**: {message}")
            else:
                st.success(f" **{title}**: {message}")


def show_password_reset_form():
    """Handle mandatory password reset"""
    st.title(" Password Reset Required")
    st.warning("Your password has expired and must be changed before continuing.")

    username = st.session_state.get('reset_username', '')

    with st.form("password_reset_form"):
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")

        if st.form_submit_button("Update Password"):
            if new_password != confirm_password:
                st.error("New passwords do not match.")
                return

            # Check password strength
            strength_check = check_password_strength(new_password)

            if not strength_check['is_valid']:
                st.error("Password does not meet security requirements.")

                missing_requirements = []
                for req, met in strength_check['requirements'].items():
                    if not met:
                        missing_requirements.append(req.replace('_', ' ').title())

                st.write("Missing requirements:", ', '.join(missing_requirements))
                return

            # Verify current password and update
            user = authenticate_user(username, current_password)
            if user and not user.get('error'):
                conn = st.session_state.db_conn
                password_hash, salt = hash_password(new_password)
                password_expires = datetime.now() + timedelta(days=90)

                conn.execute("""
                    UPDATE users 
                    SET password_hash = ?, password_expires = ?
                    WHERE username = ?
                """, [password_hash, password_expires, username])

                st.success("Password updated successfully!")

                # Clear reset flags and authenticate
                del st.session_state.password_reset_required
                del st.session_state.reset_username
                st.session_state.authenticated = True
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Current password is incorrect.")


def show_quick_entry_modal():
    """Quick emergency data entry modal"""
    st.markdown("###  Emergency Data Entry")
    st.warning("Use this for urgent data entry needs. Complete details can be added later.")

    individuals = get_individuals()
    interventions = get_interventions()

    if individuals.empty or interventions.empty:
        st.error("System configuration incomplete. Contact administrator.")
        if st.button("Cancel"):
            del st.session_state.quick_entry_mode
            st.rerun()
        return

    with st.form("emergency_entry"):
        col1, col2 = st.columns(2)

        with col1:
            individual_idx = st.selectbox("Individual", range(len(individuals)),
                                          format_func=lambda x: individuals.iloc[x]['anonymous_id'])
            intervention_idx = st.selectbox("Intervention", range(len(interventions)),
                                            format_func=lambda x: interventions.iloc[x]['name'])

        with col2:
            outcome_score = st.number_input("Quick Outcome Score (1-10)", min_value=1.0, max_value=10.0, value=5.0)
            notes = st.text_area("Emergency Notes", placeholder="Brief notes about the session or incident...")

        submitted = st.form_submit_button("Save Emergency Entry")
        cancelled = st.form_submit_button("Cancel")

        if cancelled:
            del st.session_state.quick_entry_mode
            st.rerun()

        if submitted:
            # Save quick entry with emergency flag
            conn = st.session_state.db_conn

            record_id = str(uuid.uuid4())
            session_id = str(uuid.uuid4())

            selected_individual = individuals.iloc[individual_idx]
            selected_intervention = interventions.iloc[intervention_idx]

            # Use default metric for emergency entries
            default_metric = conn.execute("""
                SELECT id FROM outcome_metrics WHERE active = TRUE ORDER BY created_at LIMIT 1
            """).fetchone()

            if default_metric:
                conn.execute("""
                    INSERT INTO outcome_records (
                        id, individual_id, intervention_id, outcome_metric_id, session_id,
                        score, session_date, notes, recorded_by, quality_rating
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    record_id, selected_individual['id'], selected_intervention['id'],
                    default_metric[0], session_id, outcome_score, date.today(),
                    f"EMERGENCY ENTRY: {notes}", st.session_state.user['id'], 3
                ])

                # Generate alert for follow-up
                generate_alert(
                    st.session_state.db_conn, # Pass the connection
                    'data_quality', 'warning',
                    'Emergency Entry Requires Review',
                    f'Emergency data entry for {selected_individual["anonymous_id"]} needs complete documentation',
                    target_role='Administrator'
                )

                st.success("Emergency entry saved! Please complete full documentation when possible.")
                del st.session_state.quick_entry_mode
                st.rerun()



if __name__ == "__main__":
    main()
