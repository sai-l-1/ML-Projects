# Student Performance Level Prediction

Machine Learning, Assignment 2 (M.Tech AIML, BITS Pilani WILP)

- **Live Streamlit app:** https://ml-projects-jkkp3kh3jdtxnxqqpdysuu.streamlit.app/
- **GitHub repository:** https://github.com/sai-l-1/ML-Projects/tree/ML-Assignment2

---

## a. Problem statement

Predict how well a student is likely to perform in exams (**Low**, **Medium** or **High**) from their daily habits and background, such as study hours, sleep, attendance, social media and Netflix time, diet, exercise and mental health rating. Five classification models (Logistic Regression, Decision Tree, KNN, Naive Bayes and SVM) are trained on the same data, compared on six evaluation metrics, and demonstrated in an interactive Streamlit app that evaluates the saved models on an uploaded test CSV.

This is a **multi-class** classification problem.

## b. Dataset description

- **Source:** Student Habits vs Academic Performance dataset (Kaggle): https://github.com/tfxhk/Regression-dataset/blob/main/student_habits_performance.csv
- **Instances:** 1000 students
- **Raw columns:** 16 (including the identifier and the original score)
- **Features used:** 14 
  - Numeric (8): `age`, `study_hours_per_day`, `social_media_hours`, `netflix_hours`, `attendance_percentage`, `sleep_hours`, `exercise_frequency`, `mental_health_rating`
  - Categorical (6): `gender`, `part_time_job`, `diet_quality`, `parental_education_level`, `internet_quality`, `extracurricular_participation`
- **Target:** `performance_level`, derived from the continuous `exam_score` (0 to 100):
  - Low: score below 50 (131 students)
  - Medium: score from 50 up to (not including) 70 (358 students)
  - High: score 70 or above (511 students)
- **Dropped columns:** `student_id` (identifier only) and `exam_score` (the target was derived from it, so keeping it would leak the answer).
- **Missing values:** `parental_education_level` has 91 missing entries, filled with the most frequent category inside the model pipeline. No duplicate rows were found.
- **Preprocessing (inside each saved pipeline):** median imputation and standard scaling for numeric features (scaling is skipped for the Decision Tree and Naive Bayes), most-frequent imputation and one-hot encoding for categorical features.
- **Train/test split:** 80/20 stratified, `random_state=42` (800 training rows, 200 test rows). The 200 test rows contain 102 High, 72 Medium and 26 Low and are saved as `test_data.csv`.

## c. GitHub repository link

https://github.com/sai-l-1/ML-Projects/tree/ML-Assignment2

```
project-folder/
|-- app.py                  Streamlit application
|-- train.py                data preparation, training, evaluation, model saving
|-- metrics_utils.py        shared metric calculation
|-- requirements.txt
|-- README.md
|-- test_data.csv           held-out test data (features + performance_level)
|-- student_habits_performance.csv   original dataset
|-- results/model_comparison.csv
|-- model/                  saved pipelines (.joblib) for all five models + meta.json
```

## d. Models used

Precision, Recall and F1 are **macro-averaged** over the three classes. AUC is one-vs-rest macro-averaged. All values are on the 200-row held-out test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8100 | 0.9471 | 0.7716 | 0.7831 | 0.7770 | 0.6820 |
| Decision Tree | 0.7100 | 0.8446 | 0.6726 | 0.6659 | 0.6662 | 0.5223 |
| KNN | 0.6850 | 0.8354 | 0.6176 | 0.5718 | 0.5826 | 0.4528 |
| Naive Bayes | 0.7750 | 0.9050 | 0.7627 | 0.7288 | 0.7430 | 0.6167 |
| Support Vector Machine | 0.7850 | 0.9380 | 0.7591 | 0.7094 | 0.7272 | 0.6320 |

Settings: Logistic Regression (`max_iter=3000`), Decision Tree (`max_depth=5`, `min_samples_leaf=5`), KNN (`n_neighbors=9`), Gaussian Naive Bayes (`var_smoothing=1e-3`), SVM (RBF kernel, `C=1.0`, `probability=True`).

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on every metric (accuracy 0.810, AUC 0.947, MCC 0.682). This suggests the link between habits and performance level is largely close to linear, so a simple, well-regularised linear model captures most of the signal. It also generalises well on the small dataset, with precision and recall (0.772 and 0.783) well balanced across the classes. |
| Decision Tree | Moderate results (accuracy 0.710, MCC 0.522) and clearly behind the linear models. A depth-limited tree can only split on one feature at a time, so it approximates the smooth, additive effect of several habits with coarse steps. Its AUC (0.845) is also much lower than the top models, because a shallow tree gives coarse probability estimates. |
| KNN | Weakest model (accuracy 0.685, macro recall 0.572, MCC 0.453). Distance-based methods suffer when many features are weakly informative, and one-hot columns add extra dimensions that dilute the distance. Its low macro recall shows it struggles with the smaller classes, which have fewer neighbours to vote for them. |
| Naive Bayes | Surprisingly competitive (accuracy 0.775, AUC 0.905) despite its strong independence assumption, since the habits are mostly independent of each other. It needed a larger `var_smoothing` (1e-3): with the default, near-zero variance in some one-hot columns made it overconfident and accuracy dropped to about 0.39. |
| Support Vector Machine | Second best on AUC (0.938) and close to Logistic Regression on accuracy (0.785), with MCC 0.632. The RBF kernel handles the scaled features well, but it did not beat the linear model, which points again to a mostly linear relationship. Its macro recall (0.709) is lower than Logistic Regression's (0.783). |
| **Overall winner for this dataset** | **Logistic Regression.** It has the highest accuracy, AUC, precision, recall, F1 and MCC, is the simplest of the five, and is fast and easy to interpret. Note that the test set has only 200 rows, so small gaps (for example SVM at 0.785 versus Naive Bayes at 0.775, a difference of two students) should not be over-interpreted. The gaps between Logistic Regression and the Decision Tree or KNN are large enough to be meaningful. |

## How to run locally with dataset present in the current execution folder

```bash
pip install -r requirements.txt
python train.py            # trains the five models, writes model/, test_data.csv, results/
streamlit run app.py
```

## Streamlit app features

- CSV upload of test data (must contain the 14 feature columns and `performance_level` with values Low, Medium or High); falls back to `test_data.csv` if nothing is uploaded
- Model selection dropdown
- Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and classification report
- Comparison table of all five models on the uploaded data
- "Try a single student" tab for a one-off prediction
