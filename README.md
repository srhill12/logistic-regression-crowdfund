
```markdown
# Logistic Regression Model for Crowdfunding Project Outcome Prediction

This project uses a Logistic Regression model to predict whether a crowdfunding project will successfully reach its goal. The dataset used is highly imbalanced, with significantly more projects that failed to meet their goal compared to those that succeeded.

## Dataset

The dataset is loaded from a CSV file, which includes information on the goal amount, the number of backers, and the outcome of the project:

```python
df = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m14/lesson_1/datasets/crowdfunding-data-imbalanced.csv")
df.head()
```

The dataset has the following columns:

- `goal`: The funding goal of the project.
- `backers_count`: The number of backers who supported the project.
- `outcome`: The binary target variable indicating whether the project was successful (`1`) or not (`0`).

### Imbalance in Data

The target variable (`outcome`) is imbalanced, with more projects failing to reach their goal:

- Successful (`1`): 565
- Unsuccessful (`0`): 111

```python
df['outcome'].value_counts()
```

## Model Training

### Data Preparation

We separate the features and the target variable:

```python
X = df.drop(columns=['outcome'])
y = df['outcome']
```

### Model Creation and Training

A Logistic Regression model is used to predict whether a crowdfunding project will reach its goal:

```python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X, y)
```

### Model Performance

#### Accuracy

The model achieved an accuracy of approximately 87.28%:

```python
classifier.score(X, y)
```

#### Confusion Matrix

The confusion matrix provides insights into the number of true positives, false positives, true negatives, and false negatives:

```python
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y, classifier.predict(X), labels=[1, 0]))
```

|              | Predicted Success (1) | Predicted Failure (0) |
|--------------|-----------------------|-----------------------|
| Actual Success (1) | 553                   | 12                    |
| Actual Failure (0) | 74                    | 37                    |

#### Classification Report

The classification report offers precision, recall, f1-score, and support for each class:

```python
from sklearn.metrics import classification_report

print(classification_report(y, classifier.predict(X), labels=[1, 0]))
```

- **Precision**: 0.88 (for class 1)
- **Recall**: 0.98 (for class 1)
- **F1-Score**: 0.93 (for class 1)

#### Balanced Accuracy Score

The balanced accuracy score, which takes class imbalance into account, is:

```python
from sklearn.metrics import balanced_accuracy_score

print(balanced_accuracy_score(y, classifier.predict(X)))
```

- **Balanced Accuracy Score**: 0.656

#### ROC AUC Score

The ROC AUC score, which measures the model's ability to distinguish between the classes, is calculated as follows:

```python
from sklearn.metrics import roc_auc_score

pred_probas = classifier.predict_proba(X)
pred_probas_firsts = [prob[1] for prob in pred_probas]
print(roc_auc_score(y, pred_probas_firsts))
```

- **ROC AUC Score**: 0.860

## Conclusion

The Logistic Regression model performed reasonably well with an accuracy of 87.28% and a strong ROC AUC score of 0.860. However, the model shows a lower balanced accuracy score, indicating some challenges in accurately predicting the minority class (`0`). The model is better at predicting successful projects than unsuccessful ones, which is typical given the imbalanced nature of the dataset.

## Future Work
Like one of my previous Logistic Regression models that had an imbalance, some of the following may be implemented in the future:
- **Resampling Techniques**: Applying oversampling techniques (e.g., SMOTE) or undersampling techniques could help improve the model's performance on the minority class.
- **Feature Engineering**: Additional features or transformations of existing features might help the model capture more nuanced patterns.
- **Model Tuning and Testing**: Hyperparameter tuning or experimenting with other classification models, such as Random Forest or XGBoost, could lead to better performance.
```
