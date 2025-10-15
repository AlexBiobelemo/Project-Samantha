import duckdb
import uuid
from datetime import datetime, date, timedelta
import random
import streamlit as st # Assuming st.session_state.db_conn is available globally or passed
from security import hash_password # Import hash_password from security.py

@st.cache_resource
def init_database():
    """Initialize DuckDB database with comprehensive schema"""
    conn = duckdb.connect('samantha_data.db')

    # Users table with enhanced fields
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id VARCHAR PRIMARY KEY,
            username VARCHAR UNIQUE NOT NULL,
            password_hash VARCHAR NOT NULL,
            salt VARCHAR NOT NULL, -- Added salt column
            role VARCHAR NOT NULL,
            email VARCHAR,
            full_name VARCHAR,
            department VARCHAR,
            phone VARCHAR,
            last_login TIMESTAMP,
            login_attempts INTEGER DEFAULT 0,
            account_locked BOOLEAN DEFAULT FALSE,
            password_expires TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            active BOOLEAN DEFAULT TRUE
        )
    """)

    # Facilities table with comprehensive settings
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facilities (
            id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            license_number VARCHAR,
            address TEXT,
            phone VARCHAR,
            email VARCHAR,
            administrator_name VARCHAR,
            capacity INTEGER,
            currency VARCHAR DEFAULT '$',
            timezone VARCHAR DEFAULT 'UTC',
            fiscal_year_start INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            active BOOLEAN DEFAULT TRUE
        )
    """)

    # Enhanced interventions table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS interventions (
            id VARCHAR PRIMARY KEY,
            facility_id VARCHAR,
            name VARCHAR NOT NULL,
            category VARCHAR,
            cost_per_session DECIMAL(10,2),
            duration_minutes INTEGER,
            group_size INTEGER DEFAULT 1,
            staff_required INTEGER DEFAULT 1,
            equipment_cost DECIMAL(10,2) DEFAULT 0,
            material_cost_per_session DECIMAL(8,2) DEFAULT 0,
            description TEXT,
            evidence_level VARCHAR DEFAULT 'Unknown',
            target_population TEXT,
            contraindications TEXT,
            created_by VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            active BOOLEAN DEFAULT TRUE
        )
    """)

    # Enhanced outcome metrics with validation
    conn.execute("""
        CREATE TABLE IF NOT EXISTS outcome_metrics (
            id VARCHAR PRIMARY KEY,
            facility_id VARCHAR,
            name VARCHAR NOT NULL,
            category VARCHAR,
            scale_min DECIMAL(8,2) DEFAULT 0,
            scale_max DECIMAL(8,2) DEFAULT 10,
            scale_type VARCHAR DEFAULT 'continuous',
            unit_of_measure VARCHAR,
            higher_is_better BOOLEAN DEFAULT TRUE,
            clinical_significance_threshold DECIMAL(8,2),
            description TEXT,
            measurement_frequency VARCHAR DEFAULT 'per_session',
            data_source VARCHAR,
            created_by VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            active BOOLEAN DEFAULT TRUE
        )
    """)

    # Enhanced individuals table with comprehensive demographics
    conn.execute("""
        CREATE TABLE IF NOT EXISTS individuals (
            id VARCHAR PRIMARY KEY,
            facility_id VARCHAR,
            anonymous_id VARCHAR NOT NULL,
            age_group VARCHAR,
            gender VARCHAR,
            disability_category VARCHAR,
            disability_severity VARCHAR,
            comorbidities TEXT,
            support_level VARCHAR,
            funding_source VARCHAR,
            admission_date DATE,
            discharge_date DATE,
            guardian_consent BOOLEAN DEFAULT TRUE,
            research_consent BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            active BOOLEAN DEFAULT TRUE
        )
    """)

    # Enhanced outcome records with session tracking
    conn.execute("""
        CREATE TABLE IF NOT EXISTS outcome_records (
            id VARCHAR PRIMARY KEY,
            individual_id VARCHAR,
            intervention_id VARCHAR,
            outcome_metric_id VARCHAR,
            session_id VARCHAR,
            score DECIMAL(8,2),
            baseline_score DECIMAL(8,2),
            target_score DECIMAL(8,2),
            session_date DATE,
            session_duration INTEGER,
            attendance_status VARCHAR DEFAULT 'attended',
            staff_id VARCHAR,
            location VARCHAR,
            weather_conditions VARCHAR,
            notes TEXT,
            quality_rating INTEGER,
            adverse_events TEXT,
            recorded_by VARCHAR,
            verified_by VARCHAR,
            verification_date TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Cost tracking table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cost_records (
            id VARCHAR PRIMARY KEY,
            intervention_id VARCHAR,
            individual_id VARCHAR,
            session_id VARCHAR,
            direct_cost DECIMAL(10,2),
            indirect_cost DECIMAL(10,2),
            overhead_cost DECIMAL(10,2),
            staff_cost DECIMAL(10,2),
            material_cost DECIMAL(10,2),
            equipment_depreciation DECIMAL(10,2),
            cost_date DATE,
            cost_category VARCHAR,
            budget_line_item VARCHAR,
            approved_by VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Goals and treatment plans
    conn.execute("""
        CREATE TABLE IF NOT EXISTS treatment_goals (
            id VARCHAR PRIMARY KEY,
            individual_id VARCHAR,
            outcome_metric_id VARCHAR,
            goal_type VARCHAR,
            baseline_value DECIMAL(8,2),
            target_value DECIMAL(8,2),
            target_date DATE,
            priority_level INTEGER,
            status VARCHAR DEFAULT 'active',
            progress_notes TEXT,
            created_by VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Audit log for data changes
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id VARCHAR PRIMARY KEY,
            table_name VARCHAR NOT NULL,
            record_id VARCHAR NOT NULL,
            action VARCHAR NOT NULL,
            old_values TEXT,
            new_values TEXT,
            changed_by VARCHAR,
            change_reason TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # System settings and configurations
    conn.execute("""
        CREATE TABLE IF NOT EXISTS system_settings (
            id VARCHAR PRIMARY KEY,
            category VARCHAR NOT NULL,
            setting_key VARCHAR NOT NULL,
            setting_value TEXT,
            description TEXT,
            data_type VARCHAR DEFAULT 'string',
            updated_by VARCHAR,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (category, setting_key)
        )
    """)

    # Add indexes for performance optimization
    conn.execute("CREATE INDEX IF NOT EXISTS idx_outcome_records_individual_id ON outcome_records (individual_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_outcome_records_intervention_id ON outcome_records (intervention_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_outcome_records_outcome_metric_id ON outcome_records (outcome_metric_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_outcome_records_session_date ON outcome_records (session_date DESC);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_individuals_anonymous_id ON individuals (anonymous_id);")

    # Reports and analytics cache
    conn.execute("""
        CREATE TABLE IF NOT EXISTS report_cache (
            id VARCHAR PRIMARY KEY,
            report_type VARCHAR NOT NULL,
            parameters TEXT,
            result_data TEXT,
            generated_by VARCHAR,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        )
    """)

    # Alerts and notifications
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id VARCHAR PRIMARY KEY,
            alert_type VARCHAR NOT NULL,
            severity VARCHAR DEFAULT 'info',
            title VARCHAR NOT NULL,
            message TEXT,
            target_user VARCHAR,
            target_role VARCHAR,
            related_entity_type VARCHAR,
            related_entity_id VARCHAR,
            is_read BOOLEAN DEFAULT FALSE,
            action_required BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        )
    """)
    
    # Table for assessment schedules
    conn.execute("""
        CREATE TABLE IF NOT EXISTS assessment_schedules (
            id VARCHAR PRIMARY KEY,
            individual_id VARCHAR NOT NULL,
            outcome_metric_id VARCHAR NOT NULL,
            frequency VARCHAR NOT NULL,
            start_date DATE NOT NULL,
            next_due_date DATE NOT NULL,
            created_by VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clinical_protocols (
            id VARCHAR PRIMARY KEY,
            name VARCHAR UNIQUE NOT NULL,
            steps TEXT,
            created_by VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Table for budget planning
    conn.execute("""
        CREATE TABLE IF NOT EXISTS budgets (
            id VARCHAR PRIMARY KEY,
            fiscal_year INTEGER UNIQUE NOT NULL,
            total_budget DECIMAL(12, 2),
            created_by VARCHAR,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Initialize with comprehensive sample data
    _initialize_sample_data(conn)

    return conn


def _initialize_sample_data(conn):
    """Initialize comprehensive sample data"""
    # Check if data already exists
    admin_exists = conn.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'").fetchone()[0]
    if admin_exists > 0:
        return

    # Create facility
    facility_id = str(uuid.uuid4())
    conn.execute("""
        INSERT INTO facilities (id, name, license_number, address, phone, administrator_name, capacity, currency)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        facility_id,
        'Samantha Care Center',
        'LIC-2024-001',
        '123 Care Drive, Healthcare City, HC 12345',
        '(555) 123-4567',
        'Dr. Sarah Johnson',
        120,
        '$'
    ])

    # Create users
    users = [
        ('admin', 'admin123', 'Administrator', 'admin@Samantha.com', 'System Administrator', 'Administration',
         '(555) 100-0001'),
        ('therapist1', 'therapy123', 'Staff', 'therapist1@Samantha.com', 'Emily Rodriguez', 'Therapy Services',
         '(555) 100-0002'),
        ('nurse1', 'nurse123', 'Staff', 'nurse1@Samantha.com', 'Michael Chen', 'Nursing', '(555) 100-0003'),
        ('supervisor1', 'super123', 'Supervisor', 'supervisor1@Samantha.com', 'Lisa Thompson', 'Clinical Services',
         '(555) 100-0004'),
    ]

    user_ids = []
    for username, password, role, email, full_name, dept, phone in users:
        user_id = str(uuid.uuid4())
        user_ids.append(user_id)
        password_hash, salt = hash_password(password) # Use the secure hash_password function and get salt
        password_expires = datetime.now() + timedelta(days=90)

        conn.execute("""
            INSERT INTO users (id, username, password_hash, salt, role, email, full_name, department, phone, password_expires)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [user_id, username, password_hash, salt, role, email, full_name, dept, phone, password_expires])

    # Create intervention categories and interventions
    intervention_data = [
        ('Physical Therapy', 'Therapy Services', 85.00, 45, 1, 1,
         'Individual physical therapy sessions focusing on mobility and strength', 'High',
         'Individuals with mobility limitations'),
        ('Speech Therapy', 'Therapy Services', 90.00, 60, 1, 1, 'Individual speech and language therapy', 'High',
         'Communication disorders'),
        ('Occupational Therapy', 'Therapy Services', 80.00, 45, 1, 1,
         'Daily living skills and adaptive equipment training', 'High', 'ADL limitations'),
        ('Group Social Skills', 'Behavioral Services', 35.00, 90, 6, 1,
         'Group-based social interaction and communication training', 'Medium', 'Social skill deficits'),
        ('Art Therapy', 'Creative Services', 65.00, 60, 4, 1,
         'Creative expression and emotional processing through art', 'Medium', 'Emotional regulation needs'),
        ('Music Therapy', 'Creative Services', 70.00, 45, 3, 1, 'Music-based therapeutic interventions', 'Medium',
         'Communication and emotional needs'),
        ('Behavioral Support', 'Behavioral Services', 95.00, 30, 1, 1, 'Individual behavioral intervention and support',
         'High', 'Challenging behaviors'),
        ('Vocational Training', 'Life Skills', 60.00, 120, 8, 1, 'Job skills and workplace readiness training',
         'Medium', 'Individuals seeking employment'),
        ('Recreational Therapy', 'Recreation', 45.00, 75, 8, 1,
         'Structured recreational activities with therapeutic goals', 'Low', 'General population'),
        ('Cognitive Training', 'Educational Services', 75.00, 45, 2, 1,
         'Cognitive skill development and memory training', 'Medium', 'Cognitive impairments'),
    ]

    intervention_ids = []
    for name, category, cost, duration, group_size, staff, description, evidence, target in intervention_data:
        intervention_id = str(uuid.uuid4())
        intervention_ids.append((intervention_id, name))

        conn.execute("""
            INSERT INTO interventions (
                id, facility_id, name, category, cost_per_session, duration_minutes, 
                group_size, staff_required, description, evidence_level, target_population,
                created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            intervention_id, facility_id, name, category, cost, duration,
            group_size, staff, description, evidence, target, user_ids[0]
        ])

    # Create outcome metrics
    outcome_metrics_data = [
        ('Social Interaction Score', 'Social Skills', 1, 10, 'continuous', 'points', True, 1.0,
         'Measures quality and frequency of social interactions'),
        ('Daily Living Skills', 'Life Skills', 0, 100, 'percentage', '%', True, 10.0,
         'Percentage of daily tasks completed independently'),
        ('Communication Effectiveness', 'Communication', 1, 5, 'ordinal', 'level', True, 1.0,
         'Level of communication effectiveness'),
        ('Behavioral Episodes', 'Behavior', 0, 20, 'count', 'episodes/week', False, 2.0,
         'Number of challenging behavioral episodes per week'),
        ('Mobility Score', 'Physical', 0, 100, 'percentage', '%', True, 15.0, 'Functional mobility assessment score'),
        ('Mood Rating', 'Emotional', 1, 10, 'continuous', 'rating', True, 2.0, 'Self-reported or observed mood rating'),
        ('Task Completion Rate', 'Cognitive', 0, 100, 'percentage', '%', True, 20.0,
         'Percentage of assigned tasks completed successfully'),
        ('Medication Compliance', 'Health', 0, 100, 'percentage', '%', True, 10.0, 'Medication adherence rate'),
        ('Sleep Quality', 'Health', 1, 10, 'continuous', 'rating', True, 1.5, 'Sleep quality rating'),
        ('Pain Level', 'Health', 0, 10, 'continuous', 'rating', False, 2.0, 'Self-reported pain level'),
    ]

    metric_ids = []
    for name, category, min_val, max_val, scale_type, unit, higher_better, threshold, description in outcome_metrics_data:
        metric_id = str(uuid.uuid4())
        metric_ids.append((metric_id, name))

        conn.execute("""
            INSERT INTO outcome_metrics (
                id, facility_id, name, category, scale_min, scale_max, scale_type,
                unit_of_measure, higher_is_better, clinical_significance_threshold,
                description, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            metric_id, facility_id, name, category, min_val, max_val, scale_type,
            unit, higher_better, threshold, description, user_ids[0]
        ])

    # Create individuals with realistic demographics
    age_groups = ['18-30', '31-45', '46-60', '60+']
    genders = ['Male', 'Female', 'Non-binary', 'Prefer not to say']
    disabilities = ['Intellectual Disability', 'Autism Spectrum Disorder', 'Down Syndrome', 'Cerebral Palsy',
                    'Traumatic Brain Injury', 'Multiple Disabilities']
    severities = ['Mild', 'Moderate', 'Severe', 'Profound']
    support_levels = ['Intermittent', 'Limited', 'Extensive', 'Pervasive']
    funding_sources = ['State Funding', 'Medicaid', 'Private Insurance', 'Private Pay', 'Mixed Funding']

    random.seed(42)  # For reproducible sample data

    individual_ids = []
    for i in range(50):  # Create 50 individuals
        individual_id = str(uuid.uuid4())
        individual_ids.append(individual_id)

        age_group = random.choice(age_groups)
        gender = random.choice(genders)
        disability = random.choice(disabilities)
        severity = random.choice(severities)
        support = random.choice(support_levels)
        funding = random.choice(funding_sources)

        # Generate admission date within last 2 years
        start_date = date(2022, 1, 1)
        end_date = date(2024, 12, 31)
        days_between = (end_date - start_date).days
        random_days = random.randint(0, days_between)
        admission_date = start_date + timedelta(days=random_days)

        conn.execute("""
            INSERT INTO individuals (
                id, facility_id, anonymous_id, age_group, gender, disability_category,
                disability_severity, support_level, funding_source, admission_date,
                guardian_consent, research_consent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            individual_id, facility_id, f'ID-{i + 1:03d}', age_group, gender, disability,
            severity, support, funding, admission_date, True, random.choice([True, False])
        ])

    # Generate comprehensive outcome data with realistic patterns
    _generate_realistic_outcome_data(conn, individual_ids, intervention_ids, metric_ids, user_ids)

    # Create treatment goals
    for individual_id in individual_ids[:20]:  # Goals for first 20 individuals
        for metric_id, metric_name in metric_ids[:5]:  # Goals for first 5 metrics
            if random.random() < 0.6:  # 60% chance of having a goal for each metric
                goal_id = str(uuid.uuid4())
                baseline = random.uniform(2, 6)
                target = min(baseline + random.uniform(1, 4), 10)
                target_date = date.today() + timedelta(days=random.randint(30, 180))

                conn.execute("""
                    INSERT INTO treatment_goals (
                        id, individual_id, outcome_metric_id, goal_type, baseline_value,
                        target_value, target_date, priority_level, created_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    goal_id, individual_id, metric_id, 'Improvement', baseline,
                    target, target_date, random.randint(1, 3), user_ids[0]
                ])

    # Initialize system settings
    system_settings = [
        ('security', 'max_login_attempts', '3', 'Maximum failed login attempts before account lock'),
        ('security', 'password_expiry_days', '90', 'Number of days before password expires'),
        ('analytics', 'statistical_significance_level', '0.05', 'P-value threshold for statistical significance'),
        ('reporting', 'default_date_range_months', '6', 'Default date range for reports in months'),
        ('alerts', 'low_attendance_threshold', '80', 'Attendance percentage below which to generate alerts'),
        ('quality', 'minimum_session_rating', '3', 'Minimum acceptable session quality rating'),
    ]

    for category, key, value, description in system_settings:
        setting_id = str(uuid.uuid4())
        conn.execute("""
            INSERT INTO system_settings (id, category, setting_key, setting_value, description, updated_by)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [setting_id, category, key, value, description, user_ids[0]])


def _generate_realistic_outcome_data(conn, individual_ids, intervention_ids, metric_ids, user_ids):
    """Generate realistic outcome data with trends and patterns"""
    random.seed(42)

    # Generate data for each individual
    for individual_id in individual_ids:
        # Each individual participates in 2-5 interventions
        num_interventions = random.randint(2, 5)
        selected_interventions = random.sample(intervention_ids, num_interventions)

        for intervention_id, intervention_name in selected_interventions:
            # Each intervention tracks 3-6 outcome metrics
            num_metrics = random.randint(3, 6)
            selected_metrics = random.sample(metric_ids, num_metrics)

            # Generate 3-18 months of data
            start_date = date(2023, 1, 1) + timedelta(days=random.randint(0, 365))
            num_sessions = random.randint(10, 60)

            for metric_id, metric_name in selected_metrics:
                # Create baseline score and improvement pattern
                if 'Behavioral Episodes' in metric_name or 'Pain Level' in metric_name:
                    # Lower is better for these metrics
                    baseline_score = random.uniform(5, 9)
                    improvement_rate = random.uniform(-0.02, -0.08)  # Negative improvement (reduction)
                else:
                    # Higher is better for most metrics
                    baseline_score = random.uniform(2, 5)
                    improvement_rate = random.uniform(0.01, 0.06)  # Positive improvement

                current_score = baseline_score

                for session_num in range(num_sessions):
                    session_id = str(uuid.uuid4())
                    record_id = str(uuid.uuid4())

                    # Calculate session date
                    days_elapsed = session_num * random.randint(3, 14)  # Sessions every 3-14 days
                    session_date = start_date + timedelta(days=days_elapsed)

                    # Apply improvement trend with some noise
                    trend_improvement = improvement_rate * session_num
                    noise = random.uniform(-0.3, 0.3)
                    current_score = baseline_score + trend_improvement + noise

                    # Apply bounds based on metric
                    if 'Behavioral Episodes' in metric_name:
                        current_score = max(0, min(20, current_score))
                    elif 'Daily Living Skills' in metric_name or 'Task Completion Rate' in metric_name or 'Medication Compliance' in metric_name:
                        current_score = max(0, min(100, current_score))
                    elif 'Pain Level' in metric_name:
                        current_score = max(0, min(10, current_score))
                    else:
                        current_score = max(1, min(10, current_score))

                    # Attendance status (90% attendance rate)
                    attendance = 'attended' if random.random() < 0.9 else random.choice(['absent', 'partial'])

                    # Session quality rating (mostly good ratings)
                    quality_weights = [0.05, 0.1, 0.2, 0.35, 0.3]  # Weights for ratings 1-5
                    quality_rating = random.choices([1, 2, 3, 4, 5], weights=quality_weights)[0]

                    # Session duration (with some variation)
                    planned_duration = 45  # Default duration
                    actual_duration = planned_duration + random.randint(-10, 15)

                    # Staff assignment
                    staff_id = random.choice(user_ids[1:])  # Exclude admin

                    # Insert outcome record
                    conn.execute("""
                        INSERT INTO outcome_records (
                            id, individual_id, intervention_id, outcome_metric_id, session_id,
                            score, baseline_score, session_date, session_duration, attendance_status,
                            staff_id, quality_rating, recorded_by
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        record_id, individual_id, intervention_id, metric_id, session_id,
                        round(current_score, 2), round(baseline_score, 2), session_date,
                        actual_duration, attendance, staff_id, quality_rating, staff_id
                    ])

                    # Generate cost record
                    cost_record_id = str(uuid.uuid4())
                    base_cost = random.uniform(50, 100)
                    material_cost = random.uniform(2, 15)
                    overhead = base_cost * 0.3

                    conn.execute("""
                        INSERT INTO cost_records (
                            id, intervention_id, individual_id, session_id, direct_cost,
                            indirect_cost, overhead_cost, material_cost, cost_date
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        cost_record_id, intervention_id, individual_id, session_id,
                        base_cost, 0, overhead, material_cost, session_date
                    ])

                    # Occasionally skip sessions to create realistic gaps
                    if random.random() < 0.1:  # 10% chance to skip next session
                        continue
