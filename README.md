# Fairness in Classification and Representation Learning

## Intro

In this project, I explore the concepts of fairness in classification and representation learning, focusing on the Adult dataset for binary classification. To evaluate classifier performance, I've defined three key metrics: accuracy, reweighed accuracy, and ΔDP. These metrics have been implemented as Python functions in the `model.py` file.

## Classification Analysis

### Correlated Features

For label Y, I computed the features most correlated with the target using Pearson correlation. The ascending list of these features is shown below:

- sex Female
- sex Male
- capital gain
- relationship Own child
- hours per week
- age u30
- marital status Never married
- education num
- relationship Husband
- marital status Married spouse

Similarly, for label A, I calculated the most correlated features with Pearson correlation and sorted them:

- occupation Craft repair
- marital status Divorced
- hours per week
- occupation Adm clerical
- relationship Wife
- relationship Unmarried
- marital status Married civ spouse
- relationship Husband
- sex Female
- sex Male

### Logistic Regression Classifier

I employed a Logistic Regression classifier to predict Y. The accuracy achieved on the entire dataset was 0.85. The re-weighted accuracy for this model reached 87.58, while the ΔDP was 0.17.

After removing the 10 most correlated columns with A (as mentioned in section 2.1), the accuracy dropped to 0.84, re-weighted accuracy to 0.86, and ΔDP to 0.13. The reduction in ΔDP indicates an increase in privacy as columns were removed.

### Analysis of Predicted Y Values

Before removing the correlated columns, the female group exhibited lower average predicted Y values when trained on logistic regression. This trend continued even after removing the 10 most correlated columns, although the difference reduced.

### Relevant Features

The most correlated features with predicted Y for A = 0 are:
- capital gain
- marital status Married civ spouse
- relationship Wife

For A = 1, the most correlated features with predicted Y are:
- marital status Married civ spouse
- relationship Husband
- education num

### Logistic Regression and Neural Net Comparison

When considering logistic regression, accuracy after removing the 10 most correlated features, as well as sex Female and sex Male features, was 0.71. The re-weighted accuracy dropped to 0.63, showing the trade-off between accuracy and privacy.

Training a 6-layer Neural Net with binary cross-entropy loss and Adam optimizer resulted in an accuracy of 0.84 after 10 epochs, with a ΔDP of 0.17, similar to logistic regression.

## Private Training using Output Perturbation

### Normalized Features

After normalizing each feature x for a point in group A = a, I used the pre-processed neural net to predict Y. The accuracy reached 84.97, but ΔDP increased to 0.20, indicating a reduction in privacy.

The normalized re-weighted accuracy was exceptionally high at 99.88, with an accuracy of 99.91 for predicting the sensitive attribute. Due to our non-Gaussian distribution, sensitive attributes were predictable from the statistic.

### Un-Normalized Features

In the case of un-normalized data, the accuracy of the Neural Net for predicting A was 0.82. The un-normalized re-weighted accuracy dropped to 0.81 compared to the normalized case.

When predicting Y in the un-normalized setting, the accuracy was 84.08, and ΔDP was 0.14. Privacy decreased compared to the normalized case, but accuracy increased.

## Internal Representation and Regularization

By measuring the distance between internal representations of the network for each group (A = 0 or 1), using a 6-layer neural net with cross-entropy loss and an MMD regularizer on the final hidden layer, the accuracy was 84.30, with a ΔDP of 0.17.

Extracting features using the same neural net without the final hidden layer yielded a normalized re-weighted accuracy of 65.15 for predicting the sensitive attribute. The accuracy dropped to 69.89, offering more privacy.

### Hyperparameter Tuning

Tuning the hyperparameter α across the range [0.01, 0.1, 1, 10, 100], revealed that as α increased, privacy increased while accuracy decreased. The table below summarizes the relationship:

| α    | Accuracy | ΔDP  |
|------|----------|------|
| 0.01 | 82.56    | 0.15 |
| 0.1  | 83.07    | 0.17 |
| 1    | 83.99    | 0.15 |
| 10   | 79.46    | 0.06 |
| 100  | 79.92    | 0.03 |

The values indicate that increasing α enhances privacy at the expense of reduced accuracy.
