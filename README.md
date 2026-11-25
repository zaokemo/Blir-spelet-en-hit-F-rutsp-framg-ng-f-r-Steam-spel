# Blir spelet en hit? Förutspå framgång för Steam-spel

## Project Description

The goal of this project is to build a machine learning model that predicts whether a Steam game will become a hit using information that can be available before the game is released.

The project is developed as an individual course project for Pixel Harbor. The final goal is to create a Streamlit application where users can enter information about a planned game and receive a hit probability and recommendation.

**Dataset:** [Steam Games Dataset by FronkonGames](https://huggingface.co/datasets/FronkonGames/steam-games-dataset)

## Project Member

- Anas Abdulla (individual project)

## Dataset and Target Variable

The dataset is restricted to games released between 2006 and 2024. Each game is represented by one row.

A game is labelled as a hit when its number of positive reviews places it in the top 25% of games released in the same year. The release year is used to create the target, but it is not used as a model feature.

The data is divided into three sets:

- Training set: 60%
- Validation set: 20%
- Test set: 20%

The split uses `random_state=42`. The same prepared split is reused throughout the modelling work.

Post-release information such as review counts, positive and negative reviews, estimated owners, peak concurrent players and playtime is excluded from the model features to reduce data leakage.

## Milestone Progress

### M1 - Initial Feature Experiments

The first milestone established a baseline and tested individual features and combinations of numerical and platform-support features. A DummyClassifier was used as a simple reference model, followed by Logistic Regression.

### M2 - Data Exploration

The dataset was explored to understand its structure, feature types, missing values and possible data-quality issues. This work also helped identify which columns could be suitable for prediction before release.

### M3 - Regression and Model Development

M3 focused on developing and evaluating regression models using pre-release game information. Different features and model configurations were tested to understand how well the available information could predict game success.

### M4 - Classification Models and Evaluation

M4 focused on predicting whether a Steam game would become a hit using classification models. Logistic Regression was evaluated using the existing training, validation and test sets.

A recommendation threshold was selected using validation data to reflect a review capacity of approximately 10%. Games above the threshold were classified as LOVANDE, while games below it were classified as REKOMMENDERAS INTE.

### M5 - Preprocessing Pipeline

M5 tested additional preprocessing methods and combined them into a scikit-learn pipeline:

- Standardization of numerical features.
- One-hot encoding of release month.
- TF-IDF features from cleaned short game descriptions.
- Multi-hot encoding of game categories and genres.
- A ColumnTransformer and Logistic Regression pipeline.

Potentially post-release promotional phrases in descriptions were investigated and cleaned before TF-IDF processing. The complete pipeline was also tested with five-fold cross-validation and with a new example game supplied as a raw input row.

#### M5 Results

The five-fold cross-validation results were:

| Model | Mean F1-score | Standard deviation |
|---|---:|---:|
| M4 baseline | 0.231 | 0.008 |
| M5 pipeline | 0.480 | 0.008 |

The M5 recommendation threshold was selected on the validation set, with a target review capacity of approximately 10%.

Final evaluation on the test set:

| Metric | Result |
|---|---:|
| Threshold | 0.6016 |
| Precision | 0.747 |
| Recall | 0.288 |
| F1-score | 0.415 |
| Recommended games | 1,908 of 19,506 |

The M5 pipeline was selected for further development because it performed better than the M4 baseline in cross-validation and in the test evaluation.

### M6 - Model Comparison and Selection

M6 compared different classification models using the preprocessing pipeline developed in M5. Logistic Regression, Decision Tree and Random Forest were evaluated using five-fold stratified cross-validation.

The initial comparison showed that Logistic Regression performed best. Further experiments tested different regularization values for Logistic Regression, with `C=0.1` giving the best average cross-validation F1-score.

#### M6 Cross-validation Results

| Model | Mean F1-score | Standard deviation |
|---|---:|---:|
| Logistic Regression | 0.558 | 0.005 |
| Decision Tree | 0.489 | 0.009 |
| Random Forest | 0.535 | 0.004 |

Logistic Regression was selected as the final model because it achieved the highest mean F1-score during cross-validation.

#### Recommendation Threshold

The recommendation threshold was selected using validation data to reflect Pixel Harbor's review capacity of approximately 10%.

The selected threshold was `0.7873`. It recommended 1,950 out of 19,506 validation games.

| Metric | Result |
|---|---:|
| Threshold | 0.7873 |
| Precision | 0.724 |
| Recall | 0.287 |
| F1-score | 0.411 |
| Recommended games | 1,950 of 19,506 |

This threshold prioritizes precision and keeps the number of recommended games close to the available review capacity, rather than maximizing the F1-score.

#### Final Test Evaluation

The selected Logistic Regression model was trained using the combined training and validation data and evaluated on the test set.

| Metric | Result |
|---|---:|
| Accuracy | 0.796 |
| Precision | 0.741 |
| Recall | 0.301 |
| F1-score | 0.428 |

#### Future Games Evaluation

An additional experiment tested how the model performed on newer games released in 2023 and 2024.

The model was trained using games released up to 2022. A separate portion of the older data was used to select the recommendation threshold before evaluating the model on the newer games.

| Metric | Result |
|---|---:|
| Threshold | 0.784 |
| Accuracy | 0.775 |
| Precision | 0.674 |
| Recall | 0.188 |
| F1-score | 0.294 |

The results show that predicting hits for newer games is more challenging. The lower recall indicates that the model misses a larger proportion of hits in this future-data experiment.

#### Saved Model

The selected Logistic Regression model and its recommendation threshold are saved using Joblib:

`models/steam_hit_model.joblib`

The saved artifact contains the trained model, selected threshold and model configuration. It can be loaded later for use in the Streamlit application.

## Project Structure

- `notebooks/00_prepare_data.ipynb` - Dataset preparation and creation of the target variable and data splits.
- `notebooks/01_features_abdullan.ipynb` - M1: Initial feature experiments and baseline model.
- `notebooks/02_know_the_data.ipynb` - M2: Dataset exploration and data-quality checks.
- `notebooks/03_regression_abdullan.ipynb` - M3: Regression and model development experiments.
- `notebooks/04_classification_abdullan.ipynb` - M4: Classification models and evaluation.
- `notebooks/05_pipeline_abdullan.ipynb` - M5: Feature preprocessing, pipeline, cross-validation and evaluation.
- `notebooks/06_model_battle_abdullan.ipynb` - M6: Model comparison, threshold selection, future-data evaluation and model saving.
- `notebooks/steam_preprocessing.py` - Reusable preprocessing functions used by the M5 and M6 pipelines.
- `models/steam_hit_model.joblib` - Saved Logistic Regression model and recommendation threshold.

The dataset files are not included in this repository. They should be placed in the `data/` folder locally.

## How to Run

1. Clone or download this repository.
2. Set up a Python environment with Jupyter Notebook and the libraries used in the notebooks, including pandas, NumPy, scikit-learn, SciPy and PyArrow.
3. Obtain the Steam Games Dataset.
4. Place the original dataset file in the `data/` folder using this filename:

   `data/train-00000-of-00001.parquet`

5. Run `notebooks/00_prepare_data.ipynb` to generate the prepared training, validation and test Parquet files.
6. Make sure the prepared files are available in the `data/` folder:
   - `data/train.parquet`
   - `data/validation.parquet`
   - `data/test.parquet`
7. Keep `steam_preprocessing.py` in the `notebooks/` folder so that the M6 notebook can import the preprocessing functions.
8. Run the milestone notebooks in order, using the same prepared data split.
9. Run the M6 notebook to reproduce the model comparison and generate the saved model artifact in the `models/` folder.

## Limitations

Some dataset fields, including current price and store descriptions, may have changed after a game was released. Although the project excludes clear post-release performance metrics and checks descriptions for potentially retrospective promotional wording, the dataset may still contain information that is not a perfect snapshot of each game's pre-release state.

The test set was evaluated during both M5 and M6. Therefore, the M6 test result should be interpreted as part of the project's milestone development rather than a completely untouched first-time test evaluation.

The future-data experiment provides an additional indication of how the model performs on newer games, but it does not guarantee how well it will predict the success of games that have not yet been released.

The model predicts whether a game belongs to the top 25% of games by positive reviews within its release year. Its probability should therefore be interpreted as a model estimate based on the available dataset, not a guarantee of commercial success.

The Streamlit application and later milestones will extend this work.