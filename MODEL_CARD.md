# Model Card - Blir spelet en hit?

## 1. Model Purpose

The purpose of this project is to help Pixel Harbor estimate the potential success of a Steam game before its release. The system predicts the probability that a game becomes a hit and estimates its number of positive reviews.

A hit is defined as a game belonging to the top 25% of positive reviews within its release year.

## 2. Dataset and Training

The project uses the FronkonGames Steam Games Dataset. Games released between 2006 and 2024 were included.

Only information that can be available before release was used as model input. Features include price, supported operating systems, required age, release month, categories, genres and short descriptions.

Information that becomes available after release, such as positive reviews, negative reviews, recommendations, owners and playtime, was excluded from the input features to avoid data leakage.

The classification and regression models were trained using separate experiments and evaluated using training, validation and test data.

## 3. Model Performance

**Classification model: Logistic Regression**

The classification model predicts whether a game becomes a hit.

- Cross-validation F1-score: 0.558
- Test accuracy: 0.796
- Test precision: 0.741
- Test recall: 0.301
- Test F1-score: 0.428
- Selected decision threshold: 0.7873

**Regression model: Random Forest Regressor**

The regression model estimates the number of positive reviews. The target was transformed using log1p during training.

- Cross-validation R²: 0.2866
- Test R²: 0.2921
- Test MAE: 1.5767 on the log scale

The Random Forest Regressor performed better than the tested Linear Regression and Decision Tree models in cross-validation.

## 4. Limitations and Ethical Considerations

The predictions are estimates and should not be treated as guarantees of commercial success. The models are based on historical Steam data, so changes in player preferences, market trends and competition may affect their accuracy.

The classification model has relatively low recall, meaning that it misses many games that actually become hits. Its recommendation should therefore be used as supporting information rather than as the only basis for business decisions.

The regression model's error is measured on the log scale. The approximate review range shown in the notebook is based on the test MAE and is not a statistically guaranteed prediction interval.

The displayed game type is summarized from the genres and categories entered by the user. It is not produced by a separate game-similarity model.

The models may also reflect biases in the historical dataset, and their predictions should be interpreted with care.