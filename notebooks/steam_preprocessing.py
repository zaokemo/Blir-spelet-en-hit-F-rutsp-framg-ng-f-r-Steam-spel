
import re
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    FunctionTransformer,
    MultiLabelBinarizer,
    StandardScaler,
    OneHotEncoder
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack


# Features used by the model
numeric_features = ["price", "required_age"]
boolean_features = ["windows", "mac", "linux"]
categorical_features = ["release_month"]
text_feature = "short_description"
list_features = ["categories", "genres"]


# Remove phrases that may reveal post-release reputation
suspicious_patterns = [
    r"award[- ]winning",
    r"acclaimed",
    r"bestselling",
    r"best-selling",
    r"players love",
    r"fan favorite",
    r"highly rated"
]


def clean_description(text):
    for pattern in suspicious_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def clean_text_column(texts):
    return [
        clean_description(str(text)) if pd.notna(text) else ""
        for text in texts
    ]


# Handle categories and genres
class MultiColumnLabelBinarizer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=object)

        self.category_encoder = MultiLabelBinarizer(sparse_output=True)
        self.genre_encoder = MultiLabelBinarizer(sparse_output=True)

        self.category_encoder.fit(X[:, 0])
        self.genre_encoder.fit(X[:, 1])

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object)

        known_categories = set(self.category_encoder.classes_)
        known_genres = set(self.genre_encoder.classes_)

        categories_clean = [
            [item for item in items if item in known_categories]
            for items in X[:, 0]
        ]

        genres_clean = [
            [item for item in items if item in known_genres]
            for items in X[:, 1]
        ]

        categories = self.category_encoder.transform(categories_clean)
        genres = self.genre_encoder.transform(genres_clean)

        return hstack([categories, genres], format="csr")


# Create the complete preprocessing pipeline
def create_preprocessor():

    text_pipeline = Pipeline([
        ("clean", FunctionTransformer(
            clean_text_column,
            validate=False
        )),
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            min_df=3
        ))
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("boolean", "passthrough", boolean_features),
            ("counts", "passthrough", [
                "categories_count",
                "genres_count"
            ]),
            ("month", OneHotEncoder(handle_unknown="ignore"),
             categorical_features),
            ("text", text_pipeline, text_feature),
            ("lists", MultiColumnLabelBinarizer(), list_features)
        ],
        remainder="drop"
    )
