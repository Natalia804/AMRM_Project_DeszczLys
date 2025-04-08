import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
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
    Wyświetla sekcję charakterystyki danych:
    - Statystyki opisowe
    - Rozkład zmiennej docelowej Group
    - Przykładowy podgląd danych
    - Wykresy korelacji
    - Boxploty wybranych zmiennych
    """
    st.title("Charakterystyka Zbioru Danych")
    st.markdown("""
    Zbiór danych zawiera informacje medyczne, socjo-ekonomiczne oraz diagnozę demencji u pacjentów. 
    Dane te zostały zaczerpnięte z <a href="https://www.kaggle.com/datasets/brsdincer/alzheimer-features/data" target="_blank" style="color: #007BFF; font-weight: bold;">Kaggle</a>, bazując na badaniach wykorzystujących uczenie maszynowe do analizy demencji.
    """, unsafe_allow_html=True)

    st.subheader("Zmienna objaśniana (Group)")
    st.markdown("""
    <span style="color: #1f77b4; font-weight: bold;">Group</span>: Diagnoza choroby:
    <ul>
      <li><code style="color: #2ca02c;">Demented</code>: Osoby zdiagnozowane z demencją.</li>
      <li><code style="color: #ff7f0e;">Nondemented</code>: Osoby bez demencji.</li>
      <li><code style="color: #d62728;">Converted</code>: Osoby, które po pewnym czasie zostały zaklasyfikowane jako zdrowe.</li>
    </ul>
    """, unsafe_allow_html=True)
    
    
    st.subheader("Zmienne objaśniające")
    st.markdown("""
    1. <span style="color: #1f77b4; font-weight: bold;">is_male</span>: Zmienna określająca płeć badanego.
    2. <span style="color: #1f77b4; font-weight: bold;">Age</span>: Wiek osoby.
    3. <span style="color: #1f77b4; font-weight: bold;">EDUC (Years of Education)</span>: Liczba lat edukacji.
    4. <span style="color: #1f77b4; font-weight: bold;">SES (Socioeconomic Status)</span>: Status społeczno-ekonomiczny (1-5, gdzie 5 to najwyższy status).
    5. <span style="color: #1f77b4; font-weight: bold;">MMSE (Mini Mental State Examination)</span>: Skala oceniająca funkcje poznawcze (pamięć, uwaga, orientacja przestrzenna).
    6. <span style="color: #1f77b4; font-weight: bold;">CDR (Clinical Dementia Rating)</span>: Skala oceny zaawansowania demencji.
    7. <span style="color: #1f77b4; font-weight: bold;">eTIV (Estimated Total Intracranial Volume)</span>: Szacowana całkowita objętość wewnątrzczaszkowa.
    8. <span style="color: #1f77b4; font-weight: bold;">nWBV (Normalized Whole Brain Volume)</span>: Znormalizowana objętość całego mózgu.
    9. <span style="color: #1f77b4; font-weight: bold;">ASF (Atlas Scaling Factor)</span>: Czynnik skalowania atlasu używany do dopasowania obrazu mózgu do wzorca.
    """, unsafe_allow_html=True)
    
    # Źródła danych
    st.subheader("Źródła Danych")
    st.markdown("""
    1. <a href="https://cordis.europa.eu/article/id/428863-mind-reading-software-finds-hidden-signs-of-dementia/pl" target="_blank" style="color: #007BFF; font-weight: bold;">Cordis.europa.eu</a>  
    2. <a href="https://www.sciencedirect.com/science/article/pii/S2352914819300917?via%3Dihub" target="_blank" style="color: #007BFF; font-weight: bold;">ScienceDirect - *Machine learning in medicine: Performance calculation of dementia prediction by support vector machines (SVM)*</a>  
    3. <a href="https://www.kaggle.com/datasets/brsdincer/alzheimer-features/data" target="_blank" style="color: #007BFF; font-weight: bold;">Kaggle Dataset</a>
    """, unsafe_allow_html=True)


    st.subheader("Podgląd pierwszych wierszy zbioru danych:")
    st.dataframe(data.head())

    st.subheader("Podstawowe statystyki:")
    st.write(data.describe())

    # Brakujące dane
    st.subheader("Brakujące dane:")
    missing_values = data.isnull().sum()
    missing_percent = (missing_values / len(data)) * 100
    st.write("Braki danych w poszczególnych kolumnach (w procentach):")
    st.bar_chart(missing_percent)
    st.write("Braki widzimy na niskim poziomie.")

    # Rozkład zmiennej Group
    st.header("Dytrybucja zmiennej objaśniającej Group")
    group_distribution = data['Group'].value_counts(normalize=True)
    st.bar_chart(group_distribution)
    st.write(group_distribution)
    st.write("Będziemy w analizie głównie korzystać z Demented i Nondemented, a różnica proporcji między nimi nie jest duża.")

    # Relacja między płcią a demencją
    st.subheader("Relacja między płcią a demencją")
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
        st.write("Tabela relacji płci a demencja:")
        st.dataframe(matrix_demented_df)

        fig, ax = plt.subplots()
        matrix_demented_df.plot(kind='bar', ax=ax, color=['indigo', 'gold'])
        ax.set_ylabel("Liczba obserwacji")
        ax.set_title("Relacja: Płeć a Demencja")
        st.pyplot(fig)
        
    st.markdown("""
    Liczba pacjentów z demencją w danych jest podobna. Tutaj proporcja jest zupełnie inna niż we
    wcześniejszym wykresie. Dane okazują się próbką niereprezentatywną, ponieważ aż **2/3
    populacji dotkniętej chorobą Alzheimera to kobiety** (1), co nie odzwierciedla nasz zestaw danych.
    Natomiast więcej jest obserwacji pacjentek żeńskich (**58%** to kobiety). Co ciekawe, wiąże się to
    między innymi z faktem, że objawy **AD rozwijają się z wiekiem**, natomiast mężczyźni zwykle
    żyją mniej niż kobiety (2).\n
    (1) [Castro-Aldrete L., *Sex and gender considerations in Alzheimer’s disease: The Women’s Brain Project contribution*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10097993/) \n
    (2) [Sangha K., *Gender differences in risk factors for transition from mild cognitive impairment to Alzheimer’s disease: A CREDOS study*](https://doi.org/10.1016/J.COMPPSYCH.2015.07.002)
    """)

    # Macierz korelacji
    st.subheader("Macierz korelacji (zmienne numeryczne)")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if not numeric_data.empty:
        corr_matrix = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm",
                    mask=mask, linewidths=0.5, vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Brak zmiennych numerycznych w zbiorze danych.")

    
    st.markdown("""
    Z mapy ciepła możemy wywnioskować, że interesująca nas korelacja zachodzi pomiędzy:
    - **nWBV – Age**
    - **nWBV – MMSE**
    - **nWBV – eTIV**
    - **ASF – eTIV**

    Zmienna **nWBV** jest stworzona ze zmiennej **eTIV**, także ich korelacja nie dziwi. Z literatury
    również wynika, że korelacja pomiędzy **Age** i **nWBV** wynika z procesu obumierania tkanki w
    mózgu wraz z wiekiem. Bardzo wysoka korelacja **ASF** i **eTIV** wynika z faktu, że **ASF** jest
    indeksem utworzonym z wartości **eTIV**.
    
    Jeśli natomiast weźmiemy pod lupę korelacje zmiennych objasniających z objaśnianymi, zauważamy 
    tu znacznie wyższą bezwzględną korelację zmiennej is_demented ze zmiennymi
    MMSE, nWBV, is_male oraz SES, a także, choć znacznie niższe, korelacje zmiennych Age i MMSE z
    klasyfikatorem is_converted. Na tej podstawie można stwierdzić zależność stwierdzonych
    zachorowań od poziomu umiejętności poznawczych, znormalizowanej objętości mózgu, płci oraz statusu ekonomicznego,
    natomiast wiek oraz umiejętności poznawcze opiszemy jako istotnie skorelowane ze
    zmienną is_converted.
    """)
    

    # Boxploty dla wybranych kolumn
    st.subheader("Dystrybucja zmiennych numerycznych względem grup diagnozy")
    st.write("Interaktywny wykres pozwala na eksploracje zmiennych.")
    columns_to_plot = ['Age', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    available_columns = [col for col in columns_to_plot if col in data.columns]

    if available_columns:
        selected_column = st.selectbox("Wybierz kolumnę do wizualizacji:", available_columns)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=data, x='Group', y=selected_column, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Brak dostępnych zmiennych do wizualizacji.")

         
    interpretacja = """
    Grupa Converted ma większy zakres wieku, i osoby z tej grupy diagnostycznej wydają się być starsze, ale generlanie nie widać drastycznych różnic między gruopami. Wyniki MMSE w grupie Demented są znacząco niższe, z większą zmiennością i wartościami odstającymi, podczas gdy grupy Nondemented i Converted mają zbliżone wyniki. Grupa Nondemented charakteryzuje się najwyższym medianowym eTIV, a grupa Converted najniższym, z wartościami odstającymi w grupie Demented. Najniższy mediana nWBV obserwowana jest w grupie Demented, co wskazuje na większe zmniejszenie objętości mózgu, podczas gdy grupy Nondemented i Converted są bardziej zbliżone. Mediana ASF pozostaje podobna między grupami, ale grupa Nondemented wykazuje większy rozkład i wartości odstające, z najmniejszym zakresem w grupie Converted.
    """

    st.header("Interpretacja Wykresów")
    st.write(interpretacja)
    
    # Zapisanie przetworzonych danych do session_state
    selected_columns = ['nWBV', 'MMSE', 'eTIV', 'SES', 'is_demented', 'is_male']
    

    available_columns = [col for col in selected_columns if col in data.columns]
    if available_columns:
        st.session_state.data_selected = data[available_columns]
    else:
        st.error("Brak wymaganych kolumn do zapisania przetworzonych danych.")

def braki_outliery_section() -> None:
    """
    Wyświetla sekcję dotyczącą:
    - Usuwania/wypełniania braków danych,
    - Analizy wartości odstających,
    - Zapisu przetworzonych danych do st.session_state.
    """
    st.title("Usuwanie braków danych i analiza wartości odstających")

    # Sprawdzenie, czy przetworzone dane są dostępne
    if "data_selected" in st.session_state:
        data_selected = st.session_state.data_selected
        #st.write("Dane zostały wczytane z `session_state`.")
    else:
        st.error("Dane nie zostały przetworzone w poprzednich sekcjach.")
        st.stop()

    data_numeric = data_selected.select_dtypes(include=['float64', 'int64'])
    st.subheader("Podgląd danych przed przetwarzaniem")
    st.dataframe(data_numeric.head())

    # Analiza braków
    st.subheader("Analiza braków danych")
    missing_numeric = data_numeric.isnull().mean() * 100
    st.bar_chart(missing_numeric)

    # Obsługa braków danych
    st.subheader("Obsługa braków danych")
    method_numeric = st.radio(
        "Interaktywna obsługa danych pozwala na zobaczenie najlepszej możliwości",
        ["Usuwanie wierszy", "Wypełnianie średnią", "Wypełnianie medianą", "Wypełnianie modą"]
    )
    

    if method_numeric == "Usuwanie wierszy":
        data_numeric_cleaned = data_numeric.dropna()
    elif method_numeric == "Wypełnianie średnią":
        data_numeric_cleaned = data_numeric.fillna(data_numeric.mean())
    elif method_numeric == "Wypełnianie medianą":
        data_numeric_cleaned = data_numeric.fillna(data_numeric.median())
    else:
        data_numeric_cleaned = data_numeric.fillna(data_numeric.mode().iloc[0])
    
    st.write("Dane numeryczne po czyszczeniu:")
    st.dataframe(data_numeric_cleaned.head())

    
    st.markdown("""
    Najmiej na średnią i odchylenie standardowe na zmienne miało wpływ uzupełnianie braków modą, 
    dlatego postawiono na to rozwiązanie. warto jedna zauważyć, że wybór metody w tym przypadku nie miał dużego wpływu na wyniki.
    """)
    data_numeric_cleaned = data_numeric.fillna(data_numeric.mode().iloc[0])
    data_cleaned = data_numeric_cleaned

    # Debug: Podgląd scalonych danych
    st.write("Dane po czyszczeniu:")
    st.dataframe(data_cleaned.head())


    # Zapis do session_state
    st.session_state.data_cleaned = data_cleaned

    # Analiza outlierów
    st.subheader("Analiza wartości odstających (Z-score)")
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data_numeric_cleaned)
    data_standardized = pd.DataFrame(data_standardized, columns=data_numeric_cleaned.columns)

    outliers_summary = {
        "Kolumna": [],
        "Liczba wartości odstających (|z|>3)": []
    }

    for col in data_standardized.columns:
        outliers = detekcja_outlier_zscore(data_standardized, col, threshold=3)
        outliers_summary["Kolumna"].append(col)
        outliers_summary["Liczba wartości odstających (|z|>3)"].append(len(outliers))

    outliers_summary_df = pd.DataFrame(outliers_summary)
    st.write("Podsumowanie liczby wartości odstających w każdej kolumnie:")
    st.dataframe(outliers_summary_df)

    st.write("""
    Ze względu na naturę medyczną problemu, zdecydowano się nie usuwać wartości odstających,
    aby nie tracić potencjalnie ważnych obserwacji.
    """)


def dzielenie_section() -> None:
    """
    Wyświetla sekcję dotyczącą:
    - Podziału danych na zbiór uczący i testowy,
    - Ewentualnej standaryzacji (poza kolumnami binarnymi),
    - Zapisu wynikowych zbiorów do st.session_state.
    """
    st.title("Dzielenie na zbiór uczący i testowy")

    if "data_cleaned" not in st.session_state:
        st.error("Brak oczyszczonych danych. Upewnij się, że poprzednie sekcje zostały wykonane.")
        st.stop()

    data_cleaned = st.session_state["data_cleaned"]

    # Sprawdzamy, czy mamy kolumnę docelową
    if "is_demented" not in data_cleaned.columns:
        st.error("Kolumna docelowa 'is_demented' nie istnieje w zbiorze danych.")
        st.stop()

    X = data_cleaned.drop(columns=["is_demented"])
    y = data_cleaned["is_demented"]

    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    # Jeśli jest kolumna binarna is_male, wyłącz ją z standaryzacji
    if "is_male" in numeric_cols:
        numeric_cols = numeric_cols.drop("is_male")

    if not numeric_cols.empty:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    st.write("Podgląd X_train po standaryzacji:")
    st.dataframe(X_train.head())
    st.write("Podgląd y_train:")
    st.write(y_train.head())
    st.write(f"Liczba próbek w zbiorze uczącym: {len(X_train)}")
    st.write(f"Liczba próbek w zbiorze testowym: {len(X_test)}")

    # Zapisujemy do session_state
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

    st.table(metrics_df)

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

    st.table(metrics_df_svm)

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

    st.table(metrics_df_rf)


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