# Project Samantha - Live Demonstration Guide

This guide provides a narrative script for demonstrating the full capabilities of Project Samantha. It's structured around user personas to showcase how the application solves real-world problems for different roles within a care facility.

**Objective:** To demonstrate how Project Samantha transforms raw data into strategic insights, operational efficiency, and improved client outcomes.

---
### **Preparation:** Before the Demo

1.  Navigate to **System Administration** -> **System Maintenance**.
2.  Click the **"Reset Demo Data to Initial State"** button. This ensures a clean, predictable dataset with known data quality issues to showcase the correction tools.

---
## Scenario 1: The Administrator ("The Big Picture")

**Persona:** A facility director or administrator responsible for budgets, overall performance, and strategic planning.

### **Part 1: The 30,000-Foot View**

1.  **Log in as `admin` / `admin123`.**
2.  Start at the **Executive Dashboard**.
3.  **Narrative:** "As a director, the first thing I need is a high-level overview. The Executive Dashboard immediately gives me my Key Performance Indicators. I can see our total investment, the average outcome score we're achieving, and our cost-efficiency."
4.  Point to the **Cost vs Effectiveness Analysis** chart. "This quadrant chart is my strategic command center. I can instantly see which programs, like 'Physical Therapy', are in our 'High Outcome, Low Cost' sweet spot, and which ones, like 'Recreational Therapy', might need a review."

### **Part 2: Solving a Data Integrity Problem**

1.  Navigate to **Data Management** -> **Data Quality Monitor**.
2.  **Narrative:** "Our analytics are only as good as our data. The 'Issues Overview' tab immediately flags critical problems. We can see we have over 9,000 duplicates, which is severely skewing our numbers."
3.  Switch to the **"Interactive Correction Tool"** tab and select "Duplicate Records."
4.  **Narrative:** "Instead of manually hunting through spreadsheets, I can fix this right here. The tool has found a group of duplicate records for an individual. With one click on 'Merge & Keep First Record', the duplicates are cleaned."
5.  Click the button to demonstrate the fix. "This ensures our analytics are always based on clean, reliable data."

### **Part 3: Making a Budget Decision**

1.  Navigate to **System Administration** -> **Financial Settings** and click the **"Budget Planning"** tab.
2.  **Narrative:** "Now that my data is clean, I can confidently plan my budget for the upcoming fiscal year. I can see my Year-to-Date spending and how it tracks against the annual budget."
3.  Use the expander to **"Set or Update Annual Budget"**. Change the value and click "Save Budget."
4.  **Narrative:** "The system is fully interactive. I can set our new budget, and all future reports and analytics will automatically reflect this new target."

---
## Scenario 2: The Clinical Supervisor ("Improving Outcomes")

**Persona:** A supervisor focused on client progress, staff performance, and the effectiveness of clinical protocols.

### **Part 1: Reviewing an Individual's Progress**

1.  **Log in as `supervisor1` / `super123`.**
2.  Navigate to **Individual Management** -> **Individual Analytics**.
3.  Select an individual (e.g., `ID-001`).
4.  **Narrative:** "As a supervisor, my focus is on client outcomes. Here, I can see a complete profile for a specific individual, including their progress timeline across different metrics and interventions."
5.  Click the **"Intervention Effectiveness"** tab. "Crucially, the system analyzes which interventions are working best *for this specific person*. We can see that 'Physical Therapy' is generating the most improvement, which helps us tailor their care plan."

### **Part 2: Using Predictive Analytics for Proactive Care**

1.  Navigate to **Analytics & Reporting** -> **Executive Dashboard** (or a similar page with the predictive tool). Click the **"Predictive Insights"** tab.
2.  If the model isn't trained, click **"Train / Retrain Predictive Model"**.
3.  **Narrative:** "This is where Project Samantha becomes truly powerful. We can use our historical data to predict the future. Let's create a hypothetical session for the individual we were just reviewing."
4.  Fill out the **Outcome Prediction Tool**. Choose parameters you know might be challenging (e.g., a low-quality rating). Click **"Predict Outcome Category"**.
5.  **Narrative:** "The model predicts a **'Low'** outcome with high confidence. This is an incredible insight. It allows me, as a supervisor, to proactively intervene. I can talk to the staff member, review the care plan, and make adjustments *before* a poor outcome occurs, moving from reactive to preventative care."

---
## Scenario 3: The Staff Member ("Daily Workflow")

**Persona:** A therapist or caregiver responsible for delivering services and logging data.

1.  **Log in as `therapist1` / `therapy123`.**
2.  Start at the **My Dashboard** -> **Personal Dashboard**.
3.  **Narrative:** "For front-line staff, the system is simplified to focus on what matters most. My personal dashboard shows me my recent activity and my impact, helping me stay engaged and track my own performance."
4.  Navigate to **Data Entry** -> **Smart Data Entry**.
5.  **Narrative:** "Data entry needs to be fast and accurate. This form is designed to prevent errors. Notice that the date defaults to today and can't be set in the future. Furthermore, if I try to submit without filling in the required session notes..."
6.  Click **"Record Session"** without filling in the notes to demonstrate the server-side validation.
7.  **Narrative:** "...the system catches it and provides a clear error message. This ensures we maintain high-quality data across the entire organization."

---
This guide demonstrates that Project Samantha is not just a collection of features, but a cohesive, role-aware platform that provides tangible value to every user, from the director's office to the therapy room.
