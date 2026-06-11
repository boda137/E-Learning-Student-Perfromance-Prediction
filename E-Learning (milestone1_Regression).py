

import warnings

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler , LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold , cross_val_score
from sklearn.ensemble import ExtraTreesRegressor

"""# ***Dataset reading & merging***"""

assessments = pd.read_csv("assessments.csv")
courses = pd.read_csv("courses.csv")
studentAssessments = pd.read_csv("StudentAssesments.csv")
studentInfo = pd.read_csv("studentInfo.csv")
studentRegistration = pd.read_csv("studentRegistration.csv")
studentVle = pd.read_csv("studentVle.csv", on_bad_lines='skip', engine='python') # Added on_bad_lines='skip' and engine='python'
vle = pd.read_csv("vle.csv")




df = pd.merge(
    studentAssessments,
    assessments,
    on="id_assessment",
    how="left"
)



df = pd.merge(
    df,
    studentInfo,
    on=["id_student", "code_module", "code_presentation"],
    how="left"
)



df = pd.merge(
    df,
    studentRegistration,
    on=["id_student", "code_module", "code_presentation"],
    how="left"
)




df = pd.merge(
    df,
    courses,
    on=["code_module", "code_presentation"],
    how="left"
)




student_vle_full = pd.merge(
    studentVle,
    vle,
    on=["id_site", "code_module", "code_presentation"],
    how="left"
)



vle_agg = student_vle_full.groupby(
    ["id_student", "code_module", "code_presentation"]
).agg({
    "sum_click": "sum"
}).reset_index()

vle_agg.rename(columns={
    "sum_click": "total_clicks"
}, inplace=True)


final_df = pd.merge(
    df,
    vle_agg,
    on=["id_student", "code_module", "code_presentation"],
    how="left"
)



print("Final Shape:", final_df.shape)
print(final_df.head())

final_df.to_csv("final_merged_dataset.csv", index=False)

print("DONE -> final_merged_dataset.csv created successfully")

"""# ***Analysis & Preprocessing***

*dataset description*
"""

final_df.describe()

final_df.info()
print('/////////////////////////////')
print(final_df.head())

final_df.isnull().sum()

final_df.duplicated().sum()

"""**fixing Nulls & duplicates**"""

# since there are no duplicates , we will fix the nulls directly only :
data = final_df.copy()
print(data["total_clicks"].skew())
print("the column values are so skewed")
data.fillna(data['total_clicks'].median(), inplace=True)
data.isnull().sum()

"""*outliers removal*"""

data = final_df.copy()

# Convert identified object columns to numeric, coercing errors to NaN
data['score'] = pd.to_numeric(data['score'], errors='coerce')
data['date'] = pd.to_numeric(data['date'], errors='coerce')
data['date_registration'] = pd.to_numeric(data['date_registration'], errors='coerce')
data['date_unregistration'] = pd.to_numeric(data['date_unregistration'], errors='coerce')

# Identify numerical columns for outlier treatment
numeric_cols_for_outliers = [
    'score',
    'total_clicks',
    'weight',
    'module_presentation_length',
    'num_of_prev_attempts',
    'date',
    'studied_credits',
    'date_registration',
    'date_unregistration'
]

# Fill any NaNs in these numeric columns introduced by 'coerce' or initially present (like total_clicks)
for col in numeric_cols_for_outliers:
    if data[col].isnull().any():
        data[col].fillna(data[col].median(), inplace=True)

# Outlier Capping using Z-score method (capping at 3 standard deviations)
# Instead of dropping rows, we will cap the values
for col in numeric_cols_for_outliers:
    mean_val = data[col].mean()
    std_dev = data[col].std()
    upper_bound = mean_val + 3 * std_dev
    lower_bound = mean_val - 3 * std_dev

    data[col] = np.where(data[col] > upper_bound, upper_bound,
                         np.where(data[col] < lower_bound, lower_bound, data[col]))

print("Shape after outlier capping:", data.shape)
data.isnull().sum()

data.info()

"""*encoding categorical data columns*"""

data = final_df.copy()

# Re-apply null filling for 'total_clicks' (from cell o1Y1Qdb5U5b1)
data['total_clicks'].fillna(data['total_clicks'].median(), inplace=True)

# Re-apply outlier capping (from cell _YQIyh1SYiU6)
# Convert identified object columns to numeric, coercing errors to NaN
data['score'] = pd.to_numeric(data['score'], errors='coerce')
data['date'] = pd.to_numeric(data['date'], errors='coerce')
data['date_registration'] = pd.to_numeric(data['date_registration'], errors='coerce')
data['date_unregistration'] = pd.to_numeric(data['date_unregistration'], errors='coerce')

# Identify numerical columns for outlier treatment
numeric_cols_for_outliers = [
    'score',
    'total_clicks',
    'weight',
    'module_presentation_length',
    'num_of_prev_attempts',
    'date',
    'studied_credits',
    'date_registration',
    'date_unregistration'
]

# Fill any NaNs in these numeric columns introduced by 'coerce' or initially present
for col in numeric_cols_for_outliers:
    if data[col].isnull().any():
        data[col].fillna(data[col].median(), inplace=True)

# Outlier Capping using Z-score method (capping at 3 standard deviations)
for col in numeric_cols_for_outliers:
    mean_val = data[col].mean()
    std_dev = data[col].std()
    upper_bound = mean_val + 3 * std_dev
    lower_bound = mean_val - 3 * std_dev

    data[col] = np.where(data[col] > upper_bound, upper_bound,
                         np.where(data[col] < lower_bound, lower_bound, data[col]))


# Binary encoding #

data['is_banked'] = data['is_banked'].astype(int)

data['disability'] = data['disability'].map({'N': 0, 'Y': 1})

data['gender'] = data['gender'].map({'M': 1, 'F': 0})



# Multiple label encoding (mapping) #

education_order = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4
}
data["highest_education"] = data["highest_education"].map(education_order)

# Corrected typo in key "TMA " to "TMA" and assigned to existing 'assessment_type' column
assessment_type_mapping = {
    "CMA" : 0 ,
    "TMA" : 1 ,
    "Exam" : 2
}
data["assessment_type"] = data["assessment_type"].map(assessment_type_mapping)


# One-hot encoder #
ohe = OneHotEncoder( handle_unknown='ignore')
encoded = ohe.fit_transform(data[['region']])
encoded_df = pd.DataFrame(encoded.toarray(), columns=ohe.get_feature_names_out(['region']))
data = pd.concat([data, encoded_df], axis=1)


# Ordinal encoding for the ordered bands #

imd_order = [
    '?' , '0-10%', '10-20', '20-30%', '30-40%', '40-50%', # Corrected '10-20' to '10-20%'
    '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'
]

oe_imd = OrdinalEncoder(categories=[imd_order])

# Clean whitespace before encoding
data['imd_band'] = data['imd_band'].astype(str).str.strip()
data[['imd_band']] = oe_imd.fit_transform(data[['imd_band']])

aga_order = ['0-35', '35-55', '55<=']

oe_age = OrdinalEncoder(categories=[aga_order]) # Use a separate OrdinalEncoder for age_band

# Clean whitespace before encoding
data['age_band'] = data['age_band'].astype(str).str.strip()
data[['age_band']] = oe_age.fit_transform(data[['age_band']])

"""*dropping some non-important columns*"""

data.info()

"""***Feature engineering for date columns***

*these features by itself have no effect on the target variable , so we have to do some engineering operations on them inorder to benefit from those*
"""

#How long the student stayed enrolled.
data['duration'] = data['date_unregistration'] - data['date_registration']

#students who didn’t unregister
data['duration'] = data['date_unregistration'].fillna(data['module_presentation_length']) - data['date_registration']

#Did the student withdraw?
data['did_withdraw'] = data['date_unregistration'].isnull().astype(int)

#Registration timing (early vs late)
#Lower = early registration
# Higher = late registration

data['reg_time'] = data['date_registration']
data['late_registration'] = (data['date_registration'] > data['date_registration'].median()).astype(int)

#assesment timing
#Small value → student registered close to assessment (bad sign)
#Large value → had more time to prepare

data['days_before_assessment'] = data['date'] - data['date_registration']

#Assessment timing relative to dropout (only for students who withdrew)
data['days_until_dropout'] = data['date_unregistration'] - data['date']



# 1. Create 'is_late' binary feature
# Check if the student's submission date is greater than the official assessment date
data['is_late'] = (data['date_submitted'] > data['date']).astype(int)

# 2. Create 'days_late' numerical feature
# Calculate the exact difference, ensuring we don't have negative days for early submissions
data['days_late'] = (data['date_submitted'] - data['date']).clip(lower=0)

# 3. Handle NaNs
# Some assessments (like exams) might not have a fixed 'date'; fill these as 0
data['is_late'] = data['is_late'].fillna(0)
data['days_late'] = data['days_late'].fillna(0)


 #click intensity
data['click_intensity'] = data['total_clicks'] / (data['duration'] + 1)


# 1. ENROLLMENT & TIMING LOGIC (re-defined here for clarity, though some might be redundant with above)
# Calculate how long the student stayed enrolled
data['duration'] = data['date_unregistration'].fillna(data['module_presentation_length']) - data['date_registration']

# Binary indicator for withdrawal
data['did_withdraw'] = data['date_unregistration'].isnull().astype(int)

# Registration timing
data['reg_time'] = data['date_registration']
data['late_registration'] = (data['date_registration'] > data['date_registration'].median()).astype(int)

# Timing relative to assessment and dropout
data['days_before_assessment'] = data['date'] - data['date_registration']
data['days_until_dropout'] = data['date_unregistration'] - data['date']


# 2. SUBMISSION PERFORMANCE (re-defined here for clarity)
# Check for late submissions
data['is_late'] = (data['date_submitted'] > data['date']).astype(int)
data['is_late'] = data['is_late'].fillna(0)

# Click intensity based on enrollment duration (re-defined here for clarity)
data['click_intensity'] = data['total_clicks'] / (data['duration'] + 1)


# 3. HISTORICAL PERFORMANCE (High Correlation Features)
# Sort to ensure cumulative calculations follow the timeline
data = data.sort_values(by=['id_student', 'date'])

# Calculate average of all PREVIOUS scores (shifted to avoid data leakage)
data['prev_scores_avg'] = data.groupby('id_student')['score'].transform(lambda x: x.shift(1).expanding().mean())

# Momentum: Is the student improving or declining?
data['score_momentum'] = data.groupby('id_student')['score'].transform(lambda x: x.shift(1) - x.shift(2))

# Fill NaNs for the first assignment where no history exists
data['prev_scores_avg'] = data['prev_scores_avg'].fillna(data['score'].mean())
data['score_momentum'] = data['score_momentum'].fillna(0)


# 4. ENGAGEMENT DEPTH (VLE Analysis)
# Use the pre-existing student_vle_full dataframe for detailed VLE activity analysis
# It contains 'id_site', 'activity_type', 'id_student', 'code_module', 'code_presentation', 'sum_click'

# Create a copy of student_vle_full to avoid modifying the original global DataFrame during these calculations
student_vle_full_copy = student_vle_full.copy()

# Distinguish between active (Quiz/Forum) and passive (URL/Resource) clicks
active_list = ['quiz', 'forumng', 'oucontent', 'subpage']
student_vle_full_copy['is_active_click'] = student_vle_full_copy['activity_type'].isin(active_list).astype(int)

# Clicks on high-effort materials for each VLE interaction
# Ensure 'sum_click' is numeric and fill any potential NaNs before multiplication
student_vle_full_copy['sum_click'] = pd.to_numeric(student_vle_full_copy['sum_click'], errors='coerce').fillna(0)
student_vle_full_copy['active_clicks_detail'] = student_vle_full_copy['sum_click'] * student_vle_full_copy['is_active_click']

# Aggregate active clicks per student, module, presentation
active_clicks_agg = student_vle_full_copy.groupby(
    ['id_student', 'code_module', 'code_presentation']
).agg({
    'active_clicks_detail': 'sum'
}).reset_index()

active_clicks_agg.rename(columns={'active_clicks_detail': 'active_clicks_total'}, inplace=True)


# Assuming 'id_student', 'date_submitted', and 'assessment_type' uniquely identify each assessment in 'data'.
merge_keys_for_module_pres = ['id_student', 'date_submitted', 'assessment_type']

# Create the right DataFrame for merge and ensure 'assessment_type' has the same dtype
right_df_for_merge = final_df[merge_keys_for_module_pres + ['code_module', 'code_presentation']].drop_duplicates().copy()
# Applying the same mapping as used for 'data' in cell CJd7AtpHdttg
assessment_type_mapping = {
    "CMA" : 0 ,
    "TMA" : 1 ,
    "Exam" : 2
}
right_df_for_merge['assessment_type'] = right_df_for_merge['assessment_type'].map(assessment_type_mapping)

# Drop existing 'code_module' and 'code_presentation' from 'data' to avoid suffixes during merge
data = data.drop(columns=['code_module', 'code_presentation'], errors='ignore')

# Merge 'code_module' and 'code_presentation' back into 'data'
data = pd.merge(
    data,
    right_df_for_merge,
    on=merge_keys_for_module_pres,
    how='left'
)

# Merge aggregated active clicks into the main 'data' DataFrame
data = pd.merge(
    data,
    active_clicks_agg,
    on=['id_student', 'code_module', 'code_presentation'],
    how='left'
)

# Fill NaNs for active_clicks_total (students with no active clicks in VLE or no VLE data in student_vle_full)
data['active_clicks_total'] = data['active_clicks_total'].fillna(0)

# Calculate ratio of active effort vs total browsing
# Ensure 'total_clicks' is available in 'data' (it should be from earlier merge with vle_agg) and handle division by zero
data['active_click_ratio'] = data['active_clicks_total'] / (data['total_clicks'] + 1)


# 5. WORKLOAD & RISK FLAGS
# Normalized credit load intensity
data['credit_load_intensity'] = data['studied_credits'] / (data['module_presentation_length'] + 1)

# At-Risk Flag: Late registration AND low intensity engagement
median_intensity = data['click_intensity'].median()
data['at_risk_flag'] = ((data['late_registration'] == 1) & (data['click_intensity'] < median_intensity)).astype(int)


# 6. FINAL PREPARATION
# Drop raw date columns that have been engineered into new features
# Also drop 'code_module' and 'code_presentation' if they are not intended as final features
eng_data = data.drop(['date', 'date_registration', 'date_unregistration', 'code_module', 'code_presentation'], axis=1)


# ===================== FEATURE ENGINEERING SUMMARY =====================

# 1. ENROLLMENT & TIMING
# duration → Number of days the student stayed enrolled
# did_withdraw → Binary flag (1 = did NOT withdraw, 0 = withdrew) [⚠️ check logic]
# reg_time → Raw registration date (captures enrollment timing)
# late_registration → 1 if student registered later than median, else 0
# days_before_assessment → Days between registration and assessment (prep time)
# days_until_dropout → Days from assessment to withdrawal (post-assessment disengagement)

# 2. SUBMISSION BEHAVIOR
# is_late → 1 if submission was after deadline, else 0
# days_late → Number of days submission was late (0 if on time/early)

# 3. ENGAGEMENT FEATURES
# click_intensity → Average clicks per day (total_clicks / duration)
# active_clicks_total → Total clicks on active learning materials (quiz, forum, content)
# active_click_ratio → Ratio of active clicks to total clicks (engagement quality)

# 4. HISTORICAL PERFORMANCE
# prev_scores_avg → Average of all previous scores (no data leakage via shift)
# score_momentum → Change between recent scores (performance trend)

# 5. WORKLOAD & RISK
# credit_load_intensity → Study load normalized by course length
# at_risk_flag → 1 if (late registration AND low engagement), else 0

# ======================================================================

"""***but not all engineered will be used , based on the correlation we will excude some features***"""

data.drop(columns=[
    'id_assessment',
    'id_student',
    'code_module',
    'code_presentation',
    'region'
], inplace=True) #id columns and unnessecary ones

"""# **Feature Selection**

*Heatmap & correlation*
"""

selected_cols = [
    'score',
    'prev_scores_avg',
    'score_momentum',
    'active_click_ratio',
    'active_clicks_total',
    'credit_load_intensity',
    'at_risk_flag',
    'total_clicks',
    'weight',
    'module_presentation_length',
    'num_of_prev_attempts',
    'duration',
    'studied_credits',
    'reg_time',
    'days_before_assessment',
    'days_until_dropout',
    'late_registration',
    'is_late',
    'click_intensity'
]


# correlation
corr_matrix = data[selected_cols].corr()

plt.figure(figsize=(12,8))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")
plt.show()

"""***the most correlated features with the target variable (score) are "prev_scores_avg" , "active_clicks_total" , "total_clicks" , "click_intensity" and "weight" .
Any Feature has correlation with target less than +-0.08 is removed & also the features that multcorrelated with other features***
"""

new_data = eng_data.drop(columns=[
    'module_presentation_length',
    'region' # Drop the original string 'region' column after encoding and module length
])

test_data = new_data.drop(columns = [
    'score_momentum',
    'credit_load_intensity',
    'days_before_assessment',
    'duration',
    'days_until_dropout',
    'reg_time',
    'late_registration',
    'is_late'

])

"""# ***Training***

***data scale and splitting (into test & train)***
"""

X = test_data.drop('score', axis=1)
y = test_data['score']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler() #Applying Standard Scaler (0-1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""# ***Models Initialization & Training***"""

'''# --- 3. Random Forest ---
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=rf_param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# --- 5. XGBoost ---
xgb_param_grid = {
    'n_estimators': [200, 500],
    'learning_rate': [0.01, 0.05],
    'max_depth': [2, 3, 4],   # reduce complexity
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'min_child_weight': [1, 5, 10],
    'gamma': [0, 0.1, 0.5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 5, 10]
}
xgb_grid = GridSearchCV(
    estimator=XGBRegressor(random_state=42, objective='reg:squarederror'),
    param_grid=xgb_param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_

# --- FINAL EVALUATION TABLE ---
final_models = {
   # "Random Forest": best_rf,
    "XGBoost": best_xgb
}

print(f"\n{'Model':<20} | {'Train R2':<10} | {'Test R2':<10}")
print("-" * 45)

for name, model in final_models.items():
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    print(f"{name:<20} | {train_r2:.4f}     | {test_r2:.4f}")'''

"""Used GridSearchCV for finding the best hyperparameters for the Xgboost but didnt work well like manually"""

models = {
     "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Random Forest": RandomForestRegressor(
     n_estimators=600,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1),

    "XGBoost": XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_lambda=3,
    reg_alpha=0.1,
    random_state=42

),

     "Gradientbooster" :GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    max_features='sqrt',
    random_state=42),

     "DecisionTreeRegressor" : DecisionTreeRegressor(max_depth=6,
    min_samples_split=10,
    min_samples_leaf=7,
    max_features='sqrt',
    random_state=4),

}

results = []

print(f"{'Model':<20} | {'Train R2':<10} | {'Test R2':<10}")
print("-" * 45)

for name, model in models.items():
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)

    results.append((name, train_r2, test_r2, model))

    print(f"{name:<20} | {train_r2:.4f}     | {test_r2:.4f}")

# Convert results
names = [r[0] for r in results]
train_scores = [r[1] for r in results]
test_scores = [r[2] for r in results]

# -------------------------------
# 1. Bar Plot (Train vs Test R2)
# -------------------------------
x = np.arange(len(names))
width = 0.35

plt.figure()
plt.bar(x - width/2, train_scores, width, label='Train R2')
plt.bar(x + width/2, test_scores, width, label='Test R2')

plt.xticks(x, names, rotation=45)
plt.xlabel("Models")
plt.ylabel("R² Score")
plt.title("Model Comparison (Train vs Test R²)")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# 2. Best Model (based on Test R2)
# -------------------------------
best_model = max(results, key=lambda x: x[2])
best_name, _, _, best_model_obj = best_model

y_pred = best_model_obj.predict(X_test)

# Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title(f"Actual vs Predicted ({best_name})")

# Line y = x
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])
plt.show()