# CampusOutcome Lab

Interactive classification project for **BITS WILP Machine Learning Assignment 2**. Six models predict whether a higher-education student will **drop out**, remain **enrolled**, or **graduate**.

## a. Problem statement

Higher-education institutions lose students who leave before completing a degree. Early identification of at-risk students lets advisors intervene with counselling, financial support, or academic help.

This project treats that decision as a **3-class classification** problem:

- **Dropout** — the student leaves the programme
- **Enrolled** — the student is still in the programme
- **Graduate** — the student has completed the programme

The same 36-feature table is used to train and compare:

1. Logistic Regression
2. Decision Tree Classifier
3. k-Nearest Neighbors
4. Naive Bayes (Gaussian)
5. Random Forest (ensemble)
6. XGBoost (gradient-boosted ensemble)

Each model is scored on a stratified hold-out test set using Accuracy, AUC (one-vs-rest, macro), Precision, Recall, F1, and Matthews Correlation Coefficient (MCC). A Streamlit app lets an evaluator upload that test file, pick a model, and inspect the metrics plus a confusion matrix.

## b. Dataset description

| Item | Detail |
| --- | --- |
| Name | Predict Students' Dropout and Academic Success |
| Source | [UCI Machine Learning Repository, dataset 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) |
| Instances | 4,424 students |
| Features | 36 predictors (demographic, socio-economic, macroeconomic, and 1st/2nd semester curricular records) |
| Target | `Target` — Dropout / Enrolled / Graduate |
| Missing values | None |
| Class mix | Graduate 2,209 (50.0%), Dropout 1,421 (32.1%), Enrolled 794 (17.9%) |

Feature groups:

- **Application / demographics:** marital status, application mode and order, course, daytime/evening attendance, previous qualification and grade, nationality, admission grade, age, gender, displaced, international, special educational needs
- **Family background:** mother's and father's qualification and occupation
- **Finance / support:** debtor, tuition fees up to date, scholarship holder
- **Academic trajectory:** curricular units credited, enrolled, evaluated, approved, grade, and without evaluation for both semesters
- **Macro context:** unemployment rate, inflation rate, GDP

**Preprocessing.** Column names are stripped of stray whitespace. Features are already stored as numeric codes in the UCI file. A stratified 80/20 split (`random_state=42`) produces 3,539 training rows and 885 test rows. Logistic Regression, kNN, and Gaussian Naive Bayes sit inside a `StandardScaler` pipeline. Tree models (Decision Tree, Random Forest, XGBoost) are trained on the raw numeric matrix. The hold-out split is saved as `test_data.csv`.

**Metric notes.** Precision, Recall, and F1 use a **weighted** average so the majority Graduate class does not hide errors on Enrolled. AUC is **macro one-vs-rest**. MCC is the multi-class implementation from scikit-learn.

## c. GitHub repository link

**Repository:** _add the public GitHub URL here after you push the project_

The repository contains:

```
app.py
requirements.txt
README.md
test_data.csv
data/data.csv
model/train_models.py
model/*.joblib
model/metrics.json
model/preprocessing.joblib
```

**Live Streamlit app:** _add the Streamlit Community Cloud URL here after deploy_

### How to run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

Retrain from scratch:

```bash
python model/train_models.py
```

## d. Models used

Hold-out test set: **885 students** (stratified 20% split).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7684 | 0.8778 | 0.7500 | 0.7684 | 0.7531 | 0.6150 |
| Decision Tree | 0.7107 | 0.8237 | 0.7010 | 0.7107 | 0.7051 | 0.5237 |
| kNN | 0.7006 | 0.8046 | 0.6813 | 0.7006 | 0.6787 | 0.4978 |
| Naive Bayes | 0.6588 | 0.7893 | 0.6334 | 0.6588 | 0.6417 | 0.4279 |
| Random Forest (Ensemble) | 0.7650 | 0.8885 | 0.7480 | 0.7650 | 0.7459 | 0.6091 |
| XGBoost | 0.7706 | 0.8928 | 0.7624 | 0.7706 | 0.7641 | 0.6211 |

### Observations on model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Strong linear baseline. Accuracy 0.7684 and F1 0.7531 sit just behind XGBoost. Tuition status, scholarship holding, and approved curricular units behave like monotonic risk factors, so a multinomial linear boundary already explains most of the signal. Enrolled recall is 0.333 — weaker than XGBoost on that minority class. |
| Decision Tree | Clear drop from the linear and boosted models (accuracy 0.7107). A depth cap of 11 and leaf-size constraints stop the tree from memorising the training set, but they also leave impure leaves around the Enrolled class (F1 0.383). Useful as an interpretable baseline, not as the production choice. |
| kNN | Distance search in 36-dimensional space is a poor match for this table. Even with `k=9` and distance weighting, Enrolled recall falls to 0.220 and overall F1 is 0.6787. Integer-coded categoricals (course, occupation, qualification) also distort Euclidean neighbourhoods. |
| Naive Bayes | Weakest of the six (accuracy 0.6588, MCC 0.4279). GaussianNB assumes independent, roughly Gaussian features. First- and second-semester curricular counts are strongly correlated, and several columns are discrete codes, so the likelihood model is misspecified. Enrolled F1 collapses to 0.219. |
| Random Forest (Ensemble) | Bagging helps: accuracy 0.7650 beats the single tree by 5.4 points, and AUC 0.8885 is second only to XGBoost. Graduate recall is very high (0.937). Enrolled recall (0.308) is still the bottleneck, which sequential boosting later improves. |
| XGBoost | **Best overall on this dataset.** Wins every required metric: accuracy 0.7706, AUC 0.8928, precision 0.7624, recall 0.7706, F1 0.7641, MCC 0.6211. Stage-wise boosting recovers the hard **Enrolled** class better than bagging (recall 0.447 vs 0.308 for Random Forest and 0.333 for logistic regression). That minority-class lift is why F1 and MCC move ahead of the other ensembles. |
| **Overall winner for this dataset?** | **XGBoost.** It is the only model that leads on all six metrics. Logistic Regression is a close, cheaper runner-up if a linear, easily explained score is preferred. Across every model the **Enrolled** class remains the main error source: it is the smallest group and sits between Dropout and Graduate in feature space. |

### Model settings

| Model | Key settings |
| --- | --- |
| Logistic Regression | `StandardScaler` + `C=0.75`, `max_iter=2500`, `solver=lbfgs` |
| Decision Tree | `max_depth=11`, `min_samples_split=18`, `min_samples_leaf=7` |
| kNN | `StandardScaler` + `n_neighbors=9`, `weights=distance` |
| Naive Bayes | `StandardScaler` + `GaussianNB` |
| Random Forest | `n_estimators=180`, `max_depth=14`, `min_samples_split=10`, `min_samples_leaf=4` |
| XGBoost | `n_estimators=220`, `max_depth=5`, `learning_rate=0.07`, `subsample=0.85`, `colsample_bytree=0.80`, `objective=multi:softprob` |

All stochastic models use `random_state=42`.

## Streamlit app features

- CSV upload for test data (or automatic use of bundled `test_data.csv`)
- Model selection dropdown for all six classifiers
- Metric cards: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix and classification report
- Optional grouped-bar comparison of every model on the loaded file

## Academic integrity note

Training code lives in `model/train_models.py`. Saved estimators are the `.joblib` files in `model/`. Metrics above are computed on the hold-out split shipped as `test_data.csv`.
