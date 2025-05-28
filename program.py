import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_validate, 
    StratifiedKFold
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
import shap


# =============== FUNKCJE POMOCNICZE =============== #

def wczytaj_dane(path: str) -> pd.DataFrame:
    """
        Loads data from a CSV file and removes the 'CDR' column (if it exists).

        Parameters
        ----------
        path : str
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the loaded data (without the 'CDR' column).
    """

    data = pd.read_csv(path)
    if 'CDR' in data.columns:
        data = data.drop(columns=['CDR'])
    return data


def przygotuj_dane_kategoryczne(data: pd.DataFrame) -> pd.DataFrame:
    """
        Maps categorical columns:
        - 'M/F' -> is_male (0/1),
        - 'Group' -> is_demented, is_converted (0/1).

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with loaded data (including 'M/F' and 'Group' columns).

        Returns
        -------
        pd.DataFrame
            DataFrame with added columns: is_male, is_demented, is_converted.
    """

    if 'M/F' in data.columns:
        data['is_male'] = data['M/F'].map({'M': True, 'F': False}).astype(int)

    if 'Group' in data.columns:
        data[['is_demented', 'is_converted']] = data['Group'].apply(
            lambda x: pd.Series({
                'Demented': (1, 0),
                'Nondemented': (0, 0),
                'Converted': (0, 1)
            }.get(x, (None, None)))
        )
        # Jeśli pojawiłyby się NaN, to konwertujemy je do 0
        data['is_demented'] = data['is_demented'].fillna(0).astype(int)
        data['is_converted'] = data['is_converted'].fillna(0).astype(int)

    return data


def wykres_macierzy_konfuzji(y_true: pd.Series, 
                             y_pred: pd.Series, 
                             ax=None, 
                             labels=["Not Demented", "Demented"]) -> None:
    """
        Draws a confusion matrix using Seaborn and Matplotlib.

        Parameters
        ----------
        y_true : pd.Series
            True labels (0/1).
        y_pred : pd.Series
            Predicted labels (0/1).
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            Axis object on which the matrix should be drawn (default is None).
        labels : list of str
            Labels for the X/Y axes in the matrix (e.g., ["Not Demented", "Demented"]).
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    if ax:
        ax.set_xlabel("Predictions")
        ax.set_ylabel("Actual")
    else:
        plt.xlabel("Predictions")
        plt.ylabel("Actual")



def detekcja_outlier_zscore(data: pd.DataFrame, 
                            column: str, 
                            threshold: float = 3.0) -> pd.DataFrame:
    """
        Returns rows that are outliers in the given column based on Z-score.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with standardized data.
        column : str
            Name of the column in which to detect outliers.
        threshold : float, optional
            Z-score threshold above which values are considered outliers (default is 3).

        Returns
        -------
        pd.DataFrame
            DataFrame rows where values in the given column exceed the threshold.
    """

    if data[column].notnull().sum() > 0:
        z_scores = zscore(data[column].dropna())
        outliers_idx = np.where(np.abs(z_scores) > threshold)[0]
        return data.iloc[outliers_idx]
    return pd.DataFrame()


# =============== FUNKCJE-SEKCJE (OBSŁUGA STREAMLIT) =============== #

def wprowadzenie_section() -> None:
  
    st.title("Applied Management Research Methods Project: Analysis of Alzheimer's Disease Using Machine Learning Methods")
    st.write("###### `Course`: Applied Management Research Methods Project:")
    st.write("###### `Teacher`: Federico Mangiò")
    st.write("###### `Course code`: 165019")
    st.write("###### `Authors`: Natalia Łyś, Zuzanna Deszcz")


   # Application Title
    st.header("Dataset Analysis: *Alzheimer Feature*")
    st.write("#### Introduction")
    st.markdown("""
    <p>
    <strong>Alzheimer's disease (AD)</strong> is the most common form of dementia.  
    In Europe, it is a leading cause of loss of independence and impairment in the elderly.  
    It is estimated that <strong>10 million people</strong> are affected.  
    The dataset includes information not only about the medical conditions (explained in more detail below)  
    but also socioeconomic factors and dementia diagnosis in patients.  
    The dataset we are using comes from  
    <a href="https://www.kaggle.com/datasets/brsdincer/alzheimer-features/data" target="_blank" style="color: #007BFF; font-weight: bold;">Kaggle</a>.
    </p>
    """, unsafe_allow_html=True)



def charakterystyka_danych_section(data: pd.DataFrame) -> None:
    """
    Displays the data characterization section:
    - Descriptive statistics
    - Distribution of the target variable Group
    - Sample data preview
    - Correlation plots
    - Boxplots of selected variables
    """
    st.title("Dataset Characteristics")
    st.markdown("""
    The dataset contains medical, socio-economic information and dementia diagnosis of patients.
    This data was taken from <a href="https://www.kaggle.com/datasets/brsdincer/alzheimer-features/data" target="_blank" style="color: #007BFF; font-weight: bold;">Kaggle</a>, based on studies using machine learning to analyze dementia.
    """, unsafe_allow_html=True)

    st.subheader("Target Variable (Group)")
    st.markdown("""
    <span style="color: #1f77b4; font-weight: bold;">Group</span>: Disease diagnosis:
    <ul>
      <li><code style="color: #2ca02c;">Demented</code>: Individuals diagnosed with dementia.</li>
      <li><code style="color: #ff7f0e;">Nondemented</code>: Individuals without dementia.</li>
      <li><code style="color: #d62728;">Converted</code>: Individuals who were later classified as healthy.</li>
    </ul>
    """, unsafe_allow_html=True)
    
    
    st.subheader("Explanatory Variables")
    st.markdown("""
    1. <span style="color: #1f77b4; font-weight: bold;">is_male</span>: Variable indicating gender of the subject.
    2. <span style="color: #1f77b4; font-weight: bold;">Age</span>: Age of the person.
    3. <span style="color: #1f77b4; font-weight: bold;">EDUC (Years of Education)</span>: Number of years of education.
    4. <span style="color: #1f77b4; font-weight: bold;">SES (Socioeconomic Status)</span>: Socioeconomic status (1–5, where 5 is the highest).
    5. <span style="color: #1f77b4; font-weight: bold;">MMSE (Mini Mental State Examination)</span>: Scale assessing cognitive functions (memory, attention, spatial orientation).
    6. <span style="color: #1f77b4; font-weight: bold;">CDR (Clinical Dementia Rating)</span>: Scale assessing dementia severity.
    7. <span style="color: #1f77b4; font-weight: bold;">eTIV (Estimated Total Intracranial Volume)</span>: Estimated total intracranial volume.
    8. <span style="color: #1f77b4; font-weight: bold;">nWBV (Normalized Whole Brain Volume)</span>: Normalized whole brain volume.
    9. <span style="color: #1f77b4; font-weight: bold;">ASF (Atlas Scaling Factor)</span>: Atlas scaling factor used to fit the brain image to the template.
    """, unsafe_allow_html=True)
    
    # Data sources
    st.subheader("Data Sources")
    st.markdown("""
    1. <a href="https://cordis.europa.eu/article/id/428863-mind-reading-software-finds-hidden-signs-of-dementia/pl" target="_blank" style="color: #007BFF; font-weight: bold;">Cordis.europa.eu</a>  
    2. <a href="https://www.sciencedirect.com/science/article/pii/S2352914819300917?via%3Dihub" target="_blank" style="color: #007BFF; font-weight: bold;">ScienceDirect - *Machine learning in medicine: Performance calculation of dementia prediction by support vector machines (SVM)*</a>  
    3. <a href="https://www.kaggle.com/datasets/brsdincer/alzheimer-features/data" target="_blank" style="color: #007BFF; font-weight: bold;">Kaggle Dataset</a>
    """, unsafe_allow_html=True)

    st.subheader("Preview of the First Rows of the Dataset:")
    st.dataframe(data.head())

    st.subheader("Basic Statistics:")
    st.write(data.describe())

    # Missing data
    st.subheader("Missing Data:")
    missing_values = data.isnull().sum()
    missing_percent = (missing_values / len(data)) * 100
    st.write("Missing values in individual columns (in percent):")
    st.bar_chart(missing_percent)
    st.write("We observe a low level of missing data.")

    # Boxplots for selected columns
    st.subheader("Distribution of Numerical Variables by Diagnosis Group")
    st.write("Interactive chart allows exploration of variables.")
    columns_to_plot = ['Age', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    available_columns = [col for col in columns_to_plot if col in data.columns]

    if available_columns:
        selected_column = st.selectbox("Select column to visualize:", available_columns)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=data, x='Group', y=selected_column, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No available variables to visualize.")

         
    interpretacja = """
    The Converted group has a wider age range and its individuals appear older, but overall, there are no drastic differences between groups. MMSE scores are significantly lower in the Demented group, with higher variability and outliers, while Nondemented and Converted groups have similar scores. The Nondemented group shows the highest median eTIV, and Converted the lowest, with outliers in the Demented group. The lowest median nWBV is in the Demented group, indicating greater brain volume loss, while Nondemented and Converted groups are more similar. Median ASF is similar across groups, but the Nondemented group shows more spread and outliers, with the smallest range in the Converted group.
    """

    st.header("Interpretation of the Charts")
    st.write(interpretacja)


    # Distribution of the Group variable
    st.header("Distribution of the Target Variable Group")
    group_distribution = data['Group'].value_counts(normalize=True)
    st.bar_chart(group_distribution)
    st.write(group_distribution)
    st.write("In our analysis, we will mainly use Demented and Nondemented, and the proportion difference between them is not large.")

    # Gender vs dementia
    st.subheader("Relationship Between Gender and Dementia")
    if 'is_male' in data.columns and 'is_demented' in data.columns:
        matrix_demented = [
            [len(data[(data['is_male'] == 1) & (data['is_demented'] == 1)]),
             len(data[(data['is_male'] == 1) & (data['is_demented'] == 0)])],
            [len(data[(data['is_male'] == 0) & (data['is_demented'] == 1)]),
             len(data[(data['is_male'] == 0) & (data['is_demented'] == 0)])]
        ]
        matrix_demented_df = pd.DataFrame(
            matrix_demented,
            index=['Male', 'Female'],
            columns=['Demented', 'Not Demented']
        )
        st.write("Table: Gender vs Dementia:")
        st.dataframe(matrix_demented_df)

        fig, ax = plt.subplots()
        matrix_demented_df.plot(kind='bar', ax=ax, color=['indigo', 'gold'])
        ax.set_ylabel("Number of Observations")
        ax.set_title("Relationship: Gender vs Dementia")
        st.pyplot(fig)
        
    st.markdown("""
    The number of dementia patients in the data is similar. Here the proportion is completely different than in the earlier chart.
    The data appears unrepresentative, because **2/3 of people affected by Alzheimer’s are women** (1), which is not reflected in our dataset.
    Interestingly, there are more female patients (**58%** are women). This may be related to the fact that **AD symptoms progress with age**, while men tend to live shorter lives than women (2).\n
    (1) [Castro-Aldrete L., *Sex and gender considerations in Alzheimer’s disease: The Women’s Brain Project contribution*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10097993/) \n
    (2) [Sangha K., *Gender differences in risk factors for transition from mild cognitive impairment to Alzheimer’s disease: A CREDOS study*](https://doi.org/10.1016/J.COMPPSYCH.2015.07.002)
    """)

    # Correlation matrix
    st.subheader("Correlation Matrix (Numerical Variables)")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if not numeric_data.empty:
        corr_matrix = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm",
                    mask=mask, linewidths=0.5, vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numerical variables in the dataset.")

    
    st.markdown("""
    From the heatmap we can conclude that interesting correlations occur between:
    - **nWBV – Age**
    - **nWBV – MMSE**
    - **nWBV – eTIV**
    - **ASF – eTIV**

    The **nWBV** variable is derived from **eTIV**, so their correlation is expected. Literature also indicates that the correlation between **Age** and **nWBV** results from brain tissue atrophy with age. The very high correlation between **ASF** and **eTIV** results from **ASF** being an index derived from **eTIV**.

    When examining the correlations between explanatory and target variables, we observe higher absolute correlation of **is_demented** with MMSE, nWBV, is_male and SES, as well as weaker correlations of Age and MMSE with **is_converted**. This suggests that dementia diagnosis may be related to cognitive ability, brain volume, gender, and socioeconomic status, while age and cognitive ability are significantly correlated with **is_converted**.
    """)
    
    # Interactive scatterplot tool
    st.subheader("Explore Relationships Between Variables (Scatterplot)")
    st.markdown("Select any two numerical variables to observe their relationship.")

    numeric_columns = list(numeric_data.columns)
    if len(numeric_columns) >= 2:
        x_var = st.selectbox("Select X-axis variable:", numeric_columns, key='x_scatter')
        y_var = st.selectbox("Select Y-axis variable:", numeric_columns, key='y_scatter')

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=data, x=x_var, y=y_var, hue='Group', palette='deep', ax=ax)
        ax.set_title(f"Scatterplot: {y_var} vs {x_var}")
        st.pyplot(fig)
    else:
        st.warning("Not enough numerical columns to generate scatterplots.")

        # Scatter-like plot: numerical variable by Group
    st.subheader("Scatter Distribution of Numerical Variables by Group")
    st.markdown("This shows the distribution of selected variable across dementia groups.")

    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'Group' in data.columns and numerical_cols:
        y_variable = st.selectbox("Select a numerical variable to visualize across groups:", numerical_cols, key="scatter_by_group")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.stripplot(data=data, x='Group', y=y_variable, jitter=True, alpha=0.6, palette='Set2', ax=ax)
        ax.set_title(f"Distribution of {y_variable} by Diagnosis Group")
        ax.set_ylabel(y_variable)
        ax.set_xlabel("Diagnosis Group")
        st.pyplot(fig)
    else:
        st.warning("Group column or numerical variables are missing.")


    
    # Save selected columns to session_state
    selected_columns = ['nWBV', 'MMSE', 'eTIV', 'SES', 'is_demented', 'is_male']
    available_columns = [col for col in selected_columns if col in data.columns]
    if available_columns:
        st.session_state.data_selected = data[available_columns]
    else:
        st.error("Required columns for saving processed data are missing.")


def braki_outliery_section() -> None:
    """
        Displays the section for:
        - Handling missing data (removal/imputation),
        - Outlier analysis,
        - Saving processed data to st.session_state.
    """
    st.title("Handling Missing Data and Outlier Analysis")

    # Check if preprocessed data is available
    if "data_selected" in st.session_state:
        data_selected = st.session_state.data_selected
        # st.write("Data loaded from `session_state`.")
    else:
        st.error("Data was not processed in previous sections.")
        st.stop()

    data_numeric = data_selected.select_dtypes(include=['float64', 'int64'])
    st.subheader("Preview of data before processing")
    st.dataframe(data_numeric.head())

    # Missing data analysis
    st.subheader("Missing Data Analysis")
    missing_numeric = data_numeric.isnull().mean() * 100
    st.bar_chart(missing_numeric)

    # Handling missing data
    st.subheader("Handling Missing Values")
    method_numeric = st.radio(
        "Interactive options for handling missing data:",
        ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"]
    )

    if method_numeric == "Drop rows":
        data_numeric_cleaned = data_numeric.dropna()
    elif method_numeric == "Fill with mean":
        data_numeric_cleaned = data_numeric.fillna(data_numeric.mean())
    elif method_numeric == "Fill with median":
        data_numeric_cleaned = data_numeric.fillna(data_numeric.median())
    else:
        data_numeric_cleaned = data_numeric.fillna(data_numeric.mode().iloc[0])

    st.write("Numerical data after cleaning:")
    st.dataframe(data_numeric_cleaned.head())

    st.markdown("""
    Filling with mode had the least impact on the mean and standard deviation of the variables,  
    so this method was chosen. However, it's worth noting that the choice of method did not significantly affect the results in this case.
    """)
    data_numeric_cleaned = data_numeric.fillna(data_numeric.mode().iloc[0])
    data_cleaned = data_numeric_cleaned

    # Debug: Cleaned data preview
    st.write("Data after final cleaning:")
    st.dataframe(data_cleaned.head())

   # Save to session_state
    st.session_state.data_cleaned = data_cleaned

    # Outlier analysis
    st.subheader("Outlier Analysis (Z-score)")
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data_numeric_cleaned)
    data_standardized = pd.DataFrame(data_standardized, columns=data_numeric_cleaned.columns)

    outliers_summary = {
        "Column": [],
        "Number of Outliers (|z|>3)": []
    }

    for col in data_standardized.columns:
        outliers = detekcja_outlier_zscore(data_standardized, col, threshold=3)
        outliers_summary["Column"].append(col)
        outliers_summary["Number of Outliers (|z|>3)"].append(len(outliers))

    outliers_summary_df = pd.DataFrame(outliers_summary)
    st.write("Summary of the number of outliers in each column:")
    st.dataframe(outliers_summary_df)

    mask_no_outliers = (np.abs(data_standardized) <= 3).all(axis=1)
    

    data_no_outliers = data_numeric_cleaned[mask_no_outliers]
    
    st.subheader("Data after removing Z-score outliers")
    st.write(f"Rows removed: {len(data_numeric_cleaned) - mask_no_outliers.sum()}")
    st.dataframe(data_no_outliers.head())
    

    data_cleaned = data_no_outliers.reset_index(drop=True)
    st.session_state.data_cleaned = data_cleaned


    st.write("""
    Due to the medical nature of the problem, we were debating wheater to delete them (because they could be potentially important observations), but after investigation of plots we decided to do so. 
    """)


def dzielenie_section() -> None:
    """
        Displays the section for:
        - Splitting the data into training and testing sets,
        - Optional standardization (excluding binary columns),
        - Saving the resulting sets to st.session_state.
    """
    st.title("Train-Test Split")

    if "data_cleaned" not in st.session_state:
        st.error("Cleaned data not found. Please make sure previous sections have been completed.")
        st.stop()

    data_cleaned = st.session_state["data_cleaned"]

    # Check if target column exists
    if "is_demented" not in data_cleaned.columns:
        st.error("Target column 'is_demented' does not exist in the dataset.")
        st.stop()

    X = data_cleaned.drop(columns=["is_demented"])
    y = data_cleaned["is_demented"]

    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    # If binary column 'is_male' exists, exclude it from standardization
    if "is_male" in numeric_cols:
        numeric_cols = numeric_cols.drop("is_male")

    if not numeric_cols.empty:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    st.write("Preview of X_train after standardization:")
    st.dataframe(X_train.head())
    st.write("Preview of y_train:")
    st.write(y_train.head())
    st.write("Although the 80–20 and 90–10 train-test splits yielded comparable results, the 70–30 split initially seemed preferable because it boosted the headline metrics. Cross-validation, however, later showed that this larger test set ultimately made the model less effective.")
    st.write(f"Number of samples in training set: {len(X_train)}")
    st.write(f"Number of samples in test set: {len(X_test)}")

    # Save to session_state
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test


def metody_uczenia_section() -> None:
   
    """
        Displays the section for training different machine learning methods (Decision Tree, SVM, Random Forest):
        - Performs GridSearchCV
        - Shows the best parameters
        - Plots confusion matrices
        - Displays and compares metrics (accuracy, precision, recall, F1)
        - Optionally analyzes SHAP values
    """
    st.title("Machine Learning Methods for Dementia Identification")

    st.write("Grid search was used to optimize the model hyperparameters. The goal is to find the best combination of hyperparameter values that maximize model performance on the given dataset.")

    required_keys = ["X_train", "X_test", "y_train", "y_test"]
    if not all(k in st.session_state for k in required_keys):
        st.error("Missing data for machine learning. Make sure the previous sections have been completed.")
        st.stop()

    X_train = st.session_state["X_train"]
    X_test = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_test = st.session_state["y_test"]

    # ---------- Cross-validation helper ----------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    def cv_report(model, X, y, name: str):
        """
        Runs 5-fold StratifiedKFold on the *training* data only
        and returns a one-row DataFrame with mean scores.
        """
        cv_out = cross_validate(
            model,
            X,
            y,
            cv=skf,
            scoring={
                "ACC": "accuracy",
                "PREC": "precision",
                "REC": "recall",
                "F1": "f1",
            },
            n_jobs=-1,
            return_train_score=False,
        )
        df = (
            pd.DataFrame(cv_out)
            .drop(columns=["fit_time", "score_time"])
            .rename(columns=lambda c: c.replace("test_", ""))
            .mean()
            .to_frame(name="Mean")
            .T.assign(Model=name)
            .set_index("Model")
            .round(3)
        )
        return df

    # ========== Decision Trees ========== #
    st.header("Method 1: Decision Trees")

    st.markdown("""
    Decision Trees are an intuitive machine learning method that models decisions in a hierarchical structure.  
    We analyze parameters such as:
    - `max_depth`: The maximum depth of the decision tree, which affects the model's complexity.
    - `criterion`: The data splitting measure (`gini` or `entropy`). The choice depends on the nature of the dataset.
    """)

    param_grid_tree = {
        'max_depth': [3, 5, 7, 10],
        'criterion': ['gini', 'entropy']
    }

    # Model training
    tree_model = DecisionTreeClassifier(random_state=42)
    grid_search_tree = GridSearchCV(tree_model, param_grid_tree, cv=5, scoring='f1', n_jobs=-1)
    grid_search_tree.fit(X_train, y_train)

    # Best model
    best_tree_model = grid_search_tree.best_estimator_
    y_pred_tree = best_tree_model.predict(X_test)

    st.write("Best parameters for Decision Tree:", grid_search_tree.best_params_)

    st.subheader("Confusion Matrix – Decision Tree")
    fig, ax = plt.subplots()
    wykres_macierzy_konfuzji(y_test, y_pred_tree, ax=ax)
    st.pyplot(fig)

    tree_accuracy = accuracy_score(y_test, y_pred_tree)
    tree_precision = precision_score(y_test, y_pred_tree)
    tree_recall = recall_score(y_test, y_pred_tree)
    tree_f1 = f1_score(y_test, y_pred_tree)

  
    # Displaying the metrics
    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Value": [tree_accuracy, tree_precision, tree_recall, tree_f1]
    }
    metrics_df = pd.DataFrame(metrics_data)

    #st.table(metrics_df)

      # ---------- Cross-validation – Decision Tree ----------
    dt_cv_df = cv_report(best_tree_model, X_train, y_train, "Decision Tree")
    #st.table(dt_cv_df)


    
        # ---------- Combined table – Decision Tree ----------
    combined_dt = pd.DataFrame(
        {
            "Test set":   [tree_accuracy, tree_precision, tree_recall, tree_f1],
            "CV mean":    [
                dt_cv_df.loc["Decision Tree", "ACC"],
                dt_cv_df.loc["Decision Tree", "PREC"],
                dt_cv_df.loc["Decision Tree", "REC"],
                dt_cv_df.loc["Decision Tree", "F1"],
            ],
        },
        index=["Accuracy", "Precision", "Recall", "F1-score"],
    ).round(3)
    
    st.subheader("Decision Tree – Test vs. 5-fold Cross Validation")
    st.table(combined_dt)


    st.subheader("ROC Curve – Decision Tree")
    y_score_tree = best_tree_model.predict_proba(X_test)[:, 1]       
    fpr_tree, tpr_tree, _ = roc_curve(y_test, y_score_tree)
    roc_auc_tree = auc(fpr_tree, tpr_tree)
    
    fig, ax = plt.subplots()
    ax.plot(fpr_tree, tpr_tree, label=f"AUC = {roc_auc_tree:0.2f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC – Decision Tree")
    ax.legend(loc="lower right")
    st.pyplot(fig)


    # Visualizing the Decision Tree
    st.subheader("Decision Tree Visualization")
    fig_tree, ax_tree = plt.subplots(figsize=(15, 8))
    plot_tree(
        best_tree_model,
        feature_names=X_train.columns,
        class_names=[str(cls) for cls in best_tree_model.classes_],
        filled=True,
        ax=ax_tree
    )
    st.pyplot(fig_tree)

    
    # ========== SVM ========== #
    st.header("Method 2: Support Vector Machines (SVM)")
    st.markdown("""
    Support Vector Machines (SVM) aim to find the optimal hyperplane to separate classes.  
    Grid Search was used to find the best parameters. Parameters:
    - `C`: Regularization (controls overfitting).
    - `kernel`: Kernel type (e.g., 'linear', 'rbf').
    """)

    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
    svm_model = SVC(random_state=42, probability=True)
    grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, scoring='f1', n_jobs=-1)
    grid_search_svm.fit(X_train, y_train)

    best_svm_model = grid_search_svm.best_estimator_
    y_pred_svm = best_svm_model.predict(X_test)

    st.write("Best SVM parameters:", grid_search_svm.best_params_)

    st.subheader("Confusion Matrix – SVM")
    fig, ax = plt.subplots()
    wykres_macierzy_konfuzji(y_test, y_pred_svm, ax=ax)
    st.pyplot(fig)

    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    svm_precision = precision_score(y_test, y_pred_svm)
    svm_recall = recall_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm)

    # Displaying results
    metrics_data_svm = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Value": [svm_accuracy, svm_precision, svm_recall, svm_f1]
    }
    metrics_df_svm = pd.DataFrame(metrics_data_svm)

    #st.table(metrics_df_svm)

        # ---------- Cross-validation – SVM ----------
    svm_cv_df = cv_report(best_svm_model, X_train, y_train, "SVM")
    # st.table(svm_cv_df)

    
    # SVM
    combined_svm = pd.DataFrame(
        {
            "Test set": [svm_accuracy, svm_precision, svm_recall, svm_f1],
            "CV mean":  [
                svm_cv_df.loc["SVM", "ACC"],
                svm_cv_df.loc["SVM", "PREC"],
                svm_cv_df.loc["SVM", "REC"],
                svm_cv_df.loc["SVM", "F1"],
            ],
        },
        index=["Accuracy", "Precision", "Recall", "F1-score"],
    ).round(3)
    
    st.subheader("SVM – Test vs. 5-fold Cross-validation")
    st.table(combined_svm)


    # ---------- ROC – SVM ----------
    st.subheader("ROC Curve – SVM")
    y_score_svm = best_svm_model.predict_proba(X_test)[:, 1]
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    
    fig, ax = plt.subplots()
    ax.plot(fpr_svm, tpr_svm, label=f"AUC = {roc_auc_svm:0.2f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC – SVM")
    ax.legend(loc="lower right")
    st.pyplot(fig)


    # ========== Random Forest ========== #
    st.header("Method 3: Random Forest")
    st.markdown("""
    Random Forest is an ensemble of decision trees that creates a strong predictive model.  
    Grid Search was used to find the best parameters:
    - `n_estimators`: Number of trees in the forest.
    - `max_depth`: Maximum depth of the trees.
    """)

    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20]
    }
    rf_model = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)

    best_rf_model = grid_search_rf.best_estimator_
    y_pred_rf = best_rf_model.predict(X_test)

    st.write("Best Random Forest parameters:", grid_search_rf.best_params_)

    st.subheader("Confusion Matrix – Random Forest")
    fig, ax = plt.subplots()
    wykres_macierzy_konfuzji(y_test, y_pred_rf, ax=ax)
    st.pyplot(fig)

    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_precision = precision_score(y_test, y_pred_rf)
    rf_recall = recall_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)

    # Displaying results
    metrics_data_rf = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Value": [rf_accuracy, rf_precision, rf_recall, rf_f1]
    }
    metrics_df_rf = pd.DataFrame(metrics_data_rf)

    #st.table(metrics_df_rf)

    # ---------- Cross-validation – Random Forest ----------
    rf_cv_df = cv_report(best_rf_model, X_train, y_train, "Random Forest")
    #st.table(rf_cv_df)


    # Random Forest
    combined_rf = pd.DataFrame(
        {
            "Test set": [rf_accuracy, rf_precision, rf_recall, rf_f1],
            "CV mean":  [
                rf_cv_df.loc["Random Forest", "ACC"],
                rf_cv_df.loc["Random Forest", "PREC"],
                rf_cv_df.loc["Random Forest", "REC"],
                rf_cv_df.loc["Random Forest", "F1"],
            ],
        },
        index=["Accuracy", "Precision", "Recall", "F1-score"],
    ).round(3)
    
    st.subheader("Random Forest – Test vs. 5-fold Cross-validation")
    st.table(combined_rf)

    
    # ---------- ROC – Random Forest ----------
    st.subheader("ROC Curve – Random Forest")
    y_score_rf = best_rf_model.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    
    fig, ax = plt.subplots()
    ax.plot(fpr_rf, tpr_rf, label=f"AUC = {roc_auc_rf:0.2f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC – Random Forest")
    ax.legend(loc="lower right")
    st.pyplot(fig)



    # ========== Summary of Results ========== #
    st.subheader("Model Comparison")
    results = [
        {
            "Model": "Decision Tree",
            "Accuracy": tree_accuracy,
            "Precision": tree_precision,
            "Recall": tree_recall,
            "F1-score": tree_f1,
        },
        {
            "Model": "SVM",
            "Accuracy": svm_accuracy,
            "Precision": svm_precision,
            "Recall": svm_recall,
            "F1-score": svm_f1,
        },
        {
            "Model": "Random Forest",
            "Accuracy": rf_accuracy,
            "Precision": rf_precision,
            "Recall": rf_recall,
            "F1-score": rf_f1,
        }
    ]
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # Comparison chart
    fig, ax = plt.subplots(figsize=(8, 4))
    results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(kind="bar", ax=ax)
    plt.title("Comparison of Classification Metrics")
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # ---------- Combined ROC Curves ----------
    st.subheader("ROC Curves – Model Comparison")
    fig, ax = plt.subplots()
    ax.plot(fpr_tree, tpr_tree, label=f"Decision Tree (AUC = {roc_auc_tree:0.2f})")
    ax.plot(fpr_svm,  tpr_svm,  label=f"SVM (AUC = {roc_auc_svm:0.2f})")
    ax.plot(fpr_rf,   tpr_rf,   label=f"Random Forest (AUC = {roc_auc_rf:0.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    st.pyplot(fig)
      
    
    
    # Results Interpretation
    st.header("Results Interpretation")

    st.write("""
    Based on the results of analyzing three machine learning models — Decision Tree, SVM, and Random Forest — we can draw conclusions regarding dementia diagnosis based on the provided features.
    """)

    st.subheader("1. Accuracy")
    st.write("""
    All three models achieved the same accuracy of 85.33%. This means that 85.33% of all diagnoses (both positive and negative) were correctly classified.  
    While this indicates overall model robustness, accuracy alone is not sufficient when dealing with imbalanced datasets, which is the case here.
    """)

    st.subheader("2. Precision")
    st.write("""
    The highest precision was achieved by the Decision Tree and SVM (90.91%), meaning that most of the cases classified as “Demented” actually belonged to that group.  
    Random Forest achieved a precision of 84.62%, which is slightly lower but still acceptable.  
    This may indicate a higher number of false positives compared to the other models.
    """)

    st.subheader("3. Recall")
    st.write("""
    The recall for the Decision Tree and SVM was 68.97%. This relatively low value means that both models missed many actual cases of dementia (false negatives).  
    This is a significant issue in medical models — we would rather detect dementia where it isn’t present than miss a real case.  
    Random Forest performed better in this category (75.86%), indicating a greater ability to detect true dementia cases.  
    This is crucial in medical analysis, as missing a dementia diagnosis can have serious consequences.
    """)

    st.subheader("4. F1-score")
    st.write("""
    The F1-score considers both precision and recall, making it a more balanced measure of model performance.  
    The Decision Tree and SVM achieved an F1-score of 78.43%, indicating similar ability to balance false positives and false negatives.  
    Random Forest achieved the highest F1-score at 80%, suggesting that it best balances precision and recall, making it the most suitable choice for this analysis.
    """)

    # Conclusions
    st.header("Conclusions")
    st.write("""
    1. **Random Forest** showed the best performance in terms of balancing precision and recall (F1-score = 80%).  
    Its higher recall makes it particularly useful when it’s crucial to minimize missed cases of dementia, which is especially important in our case.

    2. **Decision Tree and SVM** achieved very similar results, particularly in precision and F1-score,  
    but their lower recall is less desirable in medical practice.
    """)

    # SHAP Analysis Example Using the Best Model (e.g., Random Forest)
    st.header("Interpretability Analysis of the Random Forest Model - SHAP Values")
    st.markdown("""
    SHAP plots (SHapley Additive exPlanations) are a tool used to understand how individual features affect the model's decisions. Here's how to interpret them and use them in practice:

    ---

    ## How to Interpret a SHAP Plot?
    - **X-axis:** Shows SHAP values, which measure the impact of a given feature on the model’s output:
    - **Right side (positive values):** The feature increases the model’s output (in our case, increases the probability of having the disease).
    - **Left side (negative values):** The feature decreases the model’s output (reduces the probability of the disease).

    - **Y-axis:** A list of features ordered by their importance in the model — in our case, MMSE is the most important.

    - **Dot color:** Indicates the feature value:
    - **Red:** High feature value — e.g., higher MMSE test score.
    - **Blue:** Low feature value — e.g., lower MMSE test score.
    """)

    # Tworzenie obiektu SHAP Explainer
    explainer = shap.KernelExplainer(best_rf_model.predict, X_train)
    # Obliczanie wartości SHAP
    shap_values = explainer.shap_values(X_test)



    plt.clf() 
    shap.summary_plot(
        shap_values,  
        X_test,
        feature_names=X_test.columns,  
        show=False
    )
    st.pyplot(plt.gcf())
    
    st.write("""
    ### 1. `MMSE`
    - **High MMSE scores (red dots):** Decrease the likelihood of dementia (SHAP < 0), meaning good cognitive test results protect against a dementia diagnosis.
    - **Low MMSE scores (blue dots):** Increase the likelihood of dementia (SHAP > 0), suggesting that cognitive decline is strongly associated with dementia diagnosis.

    ### 2. `is_male`
    - **Men (blue dots):** Have a higher probability of developing dementia.
    - **Women (red dots):** Have a lower probability of developing dementia.
    - **Conclusion:** Male gender may appear as a risk factor in this context. However, we know from research that this conclusion is inaccurate, and the sample studied is not representative.

    ### 3. `nWBV`
    - **Lower brain volume (blue dots):** Increases the risk of dementia (SHAP > 0).
    - **Higher brain volume (red dots):** Decreases the risk of dementia (SHAP < 0).
    - **Conclusion:** Brain volume loss is a significant risk indicator for dementia.

    ### 4. `eTIV`
    - **The impact of eTIV is less pronounced, but overall:**
    - Lower eTIV values may slightly increase dementia risk.
    - Higher eTIV values have a mild protective effect.
    - **Conclusion:** Monitoring eTIV can be helpful as an additional indicator.

    ### 5. `SES`
    - **Lower SES (blue dots):** Increases the risk of dementia (SHAP > 0).
    - **Higher SES (red dots):** Decreases the risk of dementia (SHAP < 0).
    - **Conclusion:** People with lower socio-economic status are more exposed to dementia risk. Support for this group could help reduce that risk.
    """)


def podsumowanie_section() -> None:
    """
        Displays the final section summarizing the project:
        - Key conclusions
        - Brief interpretation of results
    """
    st.title("Summary and Conclusions")
    st.markdown("""
    <h4>Summary:</h4>
    <ul>
        <p>The goal of the project was to explore variables and determine which ones significantly influence the risk of developing Alzheimer's disease.  
        For this purpose, machine learning models (SVM, Decision Trees, Random Forest) were applied, and the best-performing model (Random Forest)  
        was selected for its predictive capability. Variables with the strongest impact on the disease were identified and compared with findings from existing literature.</p>

        <p>The strongest effects were observed for MMSE and nWBV, while variables like is_male, eTIV, and SES showed a smaller and more varied impact.</p>

        <ul>
            <li><strong>MMSE</strong> and <strong>nWBV</strong> are the most important indicators. Low cognitive test scores and reduced brain volume clearly increase the risk of dementia.</li>
            <li><strong>Male gender</strong> and <strong>low SES</strong> are additional risk factors, suggesting the need for targeted support.</li>
            <li><strong>eTIV</strong> and other brain parameters have a moderate impact, but monitoring them can help in risk assessment.</li>
        </ul>

        <p>The results indicate that early diagnosis and interventions aimed at improving MMSE scores can significantly reduce the risk.  
        Regular MRI/CT scans for high-risk groups may help detect issues early. Special attention should be paid to health education and prevention in these groups.</p>

        <p>The model incorporates both biological and social features, suggesting that an interdisciplinary approach is essential in the assessment and prevention of dementia.</p>
    </ul>
    """, unsafe_allow_html=True)




def dokumentacja_section() -> None:

    """
        Displays the project documentation section:
        - Project purpose and scope
        - How to run it
        - Project structure
        - Example docstrings
    """
    st.title("Project Documentation")
    st.markdown("""
    <h4>Purpose and Scope</h4>
    <p>
    The goal of this project is to build an analysis of medical and socio-economic data related to Alzheimer's disease,  
    including handling missing data, standardization, training classification models, and interpreting results.
    </p>

    <h4>How to Run</h4>
    <ul>
    <p>The application is already running and available. The code is accessible in the top right corner on <strong>GitHub</strong>.</p>
    </ul>

    <h4>File Structure</h4>
    <p>
    <ul>
        <li><code>program.py</code> – the main file that runs the Streamlit application</li>
        <li><code>alzheimer_features.csv</code> – the dataset file</li>
        <li><code>requirements.txt</code> – file containing required libraries for the app to work</li>
    </ul>
    </p>

    <h4>Task Division</h4>
    <ul>
        <li><strong>Zuzanna Deszcz</strong>:
            <ul>
                <li><strong>Data Processing:</strong>
                    <ul>
                        <li>Loading data from the CSV file.</li>
                        <li>Mapping and transforming categorical data.</li>
                    </ul>
                </li>
                <li><strong>Creating Visualizations:</strong>
                    <ul>
                        <li>Generating statistical and correlation plots.</li>
                        <li>Creating confusion matrices and boxplots.</li>
                    </ul>
                </li>
                <li><strong>Introductory Sections in Streamlit:</strong>
                    <ul>
                        <li>Developing the introduction and dataset description sections in the Streamlit app.</li>
                    </ul>
                </li>
                <li><strong>Data Cleaning:</strong>
                    <ul>
                        <li>Handling missing data (e.g., removal, value imputation).</li>
                        <li>Detecting and analyzing outliers.</li>
                    </ul>
                </li>
            </ul>
        </li>
        <li><strong>Natalia Łyś</strong>:
            <ul>
                <li><strong>Data Splitting:</strong>
                    <ul>
                        <li>Splitting data into training and testing sets.</li>
                        <li>Standardizing numerical data.</li>
                    </ul>
                </li>
                <li><strong>Training Machine Learning Models:</strong>
                    <ul>
                        <li>Implementing and optimizing models such as Decision Trees, SVM, and Random Forest.</li>
                        <li>Using GridSearchCV to select the best hyperparameters.</li>
                    </ul>
                </li>
                <li><strong>Analysis and Presentation of Results:</strong>
                    <ul>
                        <li>Evaluating models using metrics (accuracy, precision, recall, F1-score).</li>
                        <li>Creating and interpreting comparison charts.</li>
                        <li>Implementing model interpretability analysis (e.g., SHAP).</li>
                    </ul>
                </li>
            </ul>
        </li>
    </ul>

    <h4>Function Documentation (Docstrings)</h4>
    <p>
    Each function includes a short description, parameters, and return values to ensure code clarity.
    </p>
    """, unsafe_allow_html=True)



# =============== MAIN PART OF THE APPLICATION=============== #

def main() -> None:
    """
        Main function of the Streamlit application.
        Responsible for creating the sidebar menu and calling the appropriate sections.
    """

    st.sidebar.title("Navigation")
    sections = [
        "Introduction",
        "Dataset Description",
        "Missing Data Removal and Outlier Analysis",
        "Train-Test Split",
        "Machine Learning Methods",
        "Summary and Conclusions",
        "Documentation"
    ]
    selected_section = st.sidebar.radio("Go to section:", sections)

    # Wczytanie danych tylko raz (jeżeli jeszcze nie ma w session_state)
    if "data" not in st.session_state:
        data_path = 'alzheimer_features.csv'  
        data = wczytaj_dane(data_path)
        data = przygotuj_dane_kategoryczne(data)
        st.session_state["data"] = data

    # Wywołanie odpowiedniej sekcji
    if selected_section == "Introduction":
        wprowadzenie_section()
    elif selected_section == "Dataset Description":
        charakterystyka_danych_section(st.session_state["data"])
    elif selected_section == "Missing Data Removal and Outlier Analysis":
        braki_outliery_section()
    elif selected_section == "Train-Test Split":
        dzielenie_section()
    elif selected_section == "Machine Learning Methods":
        metody_uczenia_section()
    elif selected_section == "Summary and Conclusions":
        podsumowanie_section()
    elif selected_section == "Documentation":
        dokumentacja_section()


if __name__ == "__main__":
    main()
