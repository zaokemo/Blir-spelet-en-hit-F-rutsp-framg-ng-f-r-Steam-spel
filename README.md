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

Final evaluation on the held-out test set:

| Metric | Result |
|---|---:|
| Threshold | 0.6016 |
| Precision | 0.747 |
| Recall | 0.288 |
| F1-score | 0.415 |
| Recommended games | 1,908 of 19,506 |

The M5 pipeline is selected for further development because it performed better than the M4 baseline in cross-validation and on the test evaluation.

## Project Structure

- `notebooks/00_prepare_data.ipynb` - Dataset preparation and creation of the target variable and data splits.
- `notebooks/01_features_abdullan.ipynb` - M1: Initial feature experiments and baseline model.
- `notebooks/02_know_the_data.ipynb` - M2: Dataset exploration and data-quality checks.
- `notebooks/03_regression_abdullan.ipynb` - M3: Regression and model development experiments.
- `notebooks/04_classification_abdullan.ipynb` - M4: Classification models and evaluation.
- `notebooks/05_pipeline_abdullan.ipynb` - M5: Feature preprocessing, pipeline, cross-validation and final evaluation.

The dataset files are not included in this repository. They should be placed in the `data/` folder locally.

## How to Run

1. Clone or download this repository.
2. Set up a Python environment with Jupyter Notebook and the libraries used in the notebooks, including pandas, NumPy, scikit-learn, SciPy and PyArrow.
3. Obtain the Steam Games Dataset and run `notebooks/00_prepare_data.ipynb` to prepare the data.
4. Make sure the prepared training, validation and test files are available in the `data/` folder.
5. Run the milestone notebooks in order, using the same prepared data split.

The M5 notebook expects the prepared Parquet files at:

- `data/train.parquet`
- `data/validation.parquet`
- `data/test.parquet`

## Limitations

Some dataset fields, including current price and store descriptions, may have changed after a game was released. Although the project excludes clear post-release performance metrics and checks descriptions for potentially retrospective promotional wording, the dataset may still contain information that is not a perfect snapshot of each game's pre-release state.

The model predicts whether a game belongs to the top 25% of games by positive reviews within its release year. Its probability should therefore be interpreted as a model estimate based on the available dataset, not a guarantee of commercial success.

The Streamlit application and later milestones will extend this work.