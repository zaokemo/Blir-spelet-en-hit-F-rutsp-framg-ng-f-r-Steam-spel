# Blir spelet en hit? Förutspå framgång för Steam-spel

## Project Description

The goal of this project is to use machine learning to predict whether a Steam game will become a hit based on information about the game.

The project uses the **Steam Games Dataset** by FronkonGames, available on Hugging Face.

**Dataset:** [Steam Games Dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset)

## Group Members

- Anas Abdulla (individual project)

## Dataset Preparation

The dataset is restricted to games released between 2006 and 2024.

A game is defined as a hit if it belongs to the top 25% of games by positive reviews within its release year.

The dataset is divided into three parts:

- Training set: 60%
- Validation set: 20%
- Test set: 20%

The split uses `random_state=42` to ensure reproducibility.

Release year is not used as a model feature because older games have had more time to accumulate reviews.

## M1 - Feature Experiments

For the first milestone, I experimented with numerical and Boolean features to predict whether a Steam game becomes a hit. I tested price, required age, Windows support, macOS support, and Linux support.

First, I created a DummyClassifier as a baseline. It achieved approximately 74.8% accuracy on the validation set but had 0% recall for hit games because it always predicted the majority class.

I then tested Logistic Regression with individual features and combinations of features. The individual features performed poorly, so I also tested combinations using balanced class weights.

The best feature combination was price and platform support (Windows, macOS, and Linux). It achieved the following validation results:

- Accuracy: 67.7%
- Precision: 35.6%
- Recall: 35.1%
- F1-score: 35.3%

## Final Model Evaluation

After comparing different feature combinations, I selected Logistic Regression using four features: price, Windows support, macOS support, and Linux support.

The model achieved the following results on the test dataset:

- Accuracy: 67.8%
- Precision: 36.6%
- Recall: 36.3%
- F1-score: 36.4%

The results were similar to the validation results, suggesting that the model performs consistently on unseen data.

Compared with the DummyClassifier, Logistic Regression had lower accuracy but was able to identify some hit games. The DummyClassifier always predicted the majority class and therefore had zero recall for hit games.

Overall, the selected model can identify some successful Steam games, but its predictive performance is still limited. More features and improvements will be needed in later milestones.

## Limitations

Some dataset values, such as current game price and platform support, may represent the current state of a game rather than information available before its release. This is a limitation when using these features to predict success before launch.

## Project Structure

- `notebooks/00_prepare_data.ipynb` - Data preparation and creation of the target variable.
- `notebooks/01_features_abdullan.ipynb` - M1: Feature selection and baseline model.
- `notebooks/02_know_the_data.ipynb` - M2: Data exploration, quality checks, and feature suitability.

The dataset files are not included in the GitHub repository.

## How to Run

Run the notebooks in the following order:

1. `00_prepare_data.ipynb` - Prepare the dataset and create the training, validation, and test splits.
2. `01_features_abdullan.ipynb` - Train and evaluate the baseline model (M1).
3. `02_know_the_data.ipynb` - Explore the dataset, investigate data quality, and evaluate feature suitability (M2).

Make sure the required dataset files are available in the `data/` folder before running the notebooks.
