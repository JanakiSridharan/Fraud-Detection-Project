Fraudulent Transaction Detection Using Machine Learning
Project Overview
This project explores the use of traditional machine learning models to
identify potentially fraudulent financial transactions.

The main challenge is the highly imbalanced target variable: fraudulent
transactions represent only a small proportion of the records
(approximately 1.4%). The project therefore focuses on evaluating how
well models identify the minority fraud class rather than relying on
accuracy alone.

The models explored were:
- Random Forest
- XGBoost
- Logistic Regression
Business Problem
Fraudulent transactions can cause financial losses. A machine learning
model can help flag transactions that may require further investigation.

However, fraud detection involves a trade-off:
- Missing fraudulent transactions can expose the business to losses.
- Incorrectly flagging legitimate transactions can create unnecessary
  reviews and inconvenience.
For this reason, precision, recall, F1-score, and the confusion matrix
are important evaluation measures.
Dataset Overview
- Approximately 39,000 records.
- The target represents two classes: legitimate transactions (0) and
  fraudulent transactions (1).
- Fraudulent transactions account for approximately 1.4% of the
  dataset.
- 3,033 duplicate records were identified.
There was no unique transaction ID available to establish whether
repeated rows represented erroneous duplicates or genuine repeated
transactions. The records were therefore retained rather than removed
automatically.

Missing values were observed in two categorical columns during the
initial data inspection.
Project Workflow
1. Data inspection and preparation
- Examined the dataset structure and feature types.
- Checked missing values and duplicate records.
- Investigated the class distribution of legitimate and fraudulent
  transactions.
- Retained the 3,033 repeated records because there was no unique
  transaction identifier to establish that they were invalid records.
2. Feature investigation
The accountAgeDays feature showed an unusually strong relationship
with the target:
- All fraudulent records had an account age of 1 day.
- No legitimate records had an account age of 1 day.
- With this feature included, the Random Forest model classified every
  record in the evaluation set correctly.
This result prompted further investigation. To assess whether the models
could identify fraud using the other available information,
accountAgeDays was excluded from the subsequent model comparisons.
3. Model development
Three traditional machine learning models were evaluated using the
reduced feature set:
- Random Forest
- XGBoost
- Logistic Regression
4. Model evaluation
Because the fraud class is rare, accuracy alone does not provide a
reliable picture of model performance.

The evaluation focused on:
- Precision: Of the transactions predicted as fraud, how many were
  actually fraudulent?
- Recall: Of all actual fraudulent transactions, how many did the
  model identify?
- F1-score: The harmonic mean of precision and recall.
- Confusion matrix: Shows correct and incorrect predictions for
  both classes.
Results
Model comparison (without accountAgeDays)
  Model                   Accuracy   Fraud precision   Fraud recall   Fraud F1-score
  Random Forest               0.87              0.03           0.21             0.05
XGBoost                     0.72              0.03       0.70         0.07
Logistic Regression         0.56              0.03       0.85             0.05

XGBoost achieved the highest fraud-class F1-score among the three models
evaluated. Logistic Regression achieved the highest fraud recall, but
also generated a large number of false-positive predictions.
Confusion matrices
The class order is [0, 1], where 0 represents legitimate
transactions and 1 represents fraudulent transactions. Rows represent
actual classes and columns represent predicted classes.

Random Forest
[[6829,  904],
 [  88,   24]]
XGBoost
[[5547, 2186],
 [  34,   78]]
Logistic Regression
[[4302, 3431],
 [  17,   95]]
Interpretation
The results illustrate the difficulty of detecting a small minority
class:
- Random Forest achieved the highest accuracy, but identified only 21%
  of actual fraudulent transactions.
- XGBoost identified 70% of fraudulent transactions and achieved the
  best fraud-class F1-score in this comparison.
- Logistic Regression identified 85% of fraudulent transactions, but
  incorrectly flagged many legitimate transactions.
Based on the fraud-class F1-score, XGBoost was the strongest candidate
among the three models evaluated. The appropriate model in a real
business setting would also depend on the cost of missed fraud cases and
false alarms.
Key Findings
1. The target variable is highly imbalanced, with fraud representing
   approximately 1.4% of records.
2. Accuracy alone can be misleading in an imbalanced classification
   problem.
3. accountAgeDays produced perfect classification in the initial
   Random Forest evaluation, so it was excluded to examine model
   performance without that feature.
4. After excluding the feature, all three models showed a trade-off
   between identifying fraud and incorrectly flagging legitimate
   transactions.
5. XGBoost achieved the highest fraud-class F1-score in the model
   comparison, while Logistic Regression achieved the highest fraud
   recall.
Limitations and Considerations
- The dataset contains a substantial number of repeated records. They
  were retained because a unique transaction ID was unavailable to
  determine whether they were invalid duplicates or genuine repeated
  transactions.
- The unusually strong relationship between accountAgeDays and fraud
  requires further validation before that feature could be trusted in
  a real-world system.
- Fraud precision was low across all three models, meaning many
  transactions flagged as fraud were actually legitimate.
- The model comparison does not by itself establish the best
  operational decision threshold or the financial cost of false
  positives and false negatives.
- Further validation on additional data would be needed before using a
  model for live transaction decisions.
Tools and Libraries
The project used a traditional machine learning workflow. The specific
libraries and implementation details should be documented alongside the
notebook used for the project.
Conclusion
This project demonstrates a machine learning approach to fraudulent
transaction detection and highlights the challenges of working with
highly imbalanced data.

The investigation of accountAgeDays showed why model results need to
be examined rather than accepted at face value. After excluding that
feature, XGBoost achieved the highest fraud-class F1-score among the
three models evaluated, while Logistic Regression achieved the highest
recall.

The results provide a useful baseline for further work. Additional
feature validation, threshold evaluation, and testing on further data
would be necessary before considering operational use.
