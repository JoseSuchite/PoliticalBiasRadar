from pathlib import Path
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

DATASET_ID = "siddharthmb/article-bias-prediction-media-splits"


def main():

    # Load dataset
    projectDirectory = Path(__file__).resolve().parent
    dataDirectory = projectDirectory / "cleaned"

    train_data = pq.read_table(
        dataDirectory / "train-00000-of-00001.parquet"
    )

    valid_data = pq.read_table(
        dataDirectory / "valid-00000-of-00001.parquet"
    )

    test_data = pq.read_table(
        dataDirectory / "test-00000-of-00001.parquet"
    )

    # Extract articles and labels
    train_articles = train_data["content"].to_pylist()
    train_labels = train_data["bias"].to_pylist()

    valid_articles = valid_data["content"].to_pylist()
    valid_labels = valid_data["bias"].to_pylist()

    test_articles = test_data["content"].to_pylist()
    test_labels = test_data["bias"].to_pylist()

    print(f"Training samples: {len(train_labels)}")
    print(f"Validation samples: {len(valid_labels)}")
    print(f"Testing samples: {len(test_labels)}")

    print(f"Classes: {set(train_labels)}")
    print(f"Columns: {train_data.column_names}")

    # Convert articles into TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    X_train = vectorizer.fit_transform(train_articles)
    X_valid = vectorizer.transform(valid_articles)
    X_test = vectorizer.transform(test_articles)

    model = MultinomialNB(alpha=1.0)

    model.fit(X_train, train_labels)

    valid_predictions = model.predict(X_valid)

    valid_acc = accuracy_score(
        valid_labels, valid_predictions
    )

    valid_f1 = f1_score(
        valid_labels, valid_predictions,
        average="macro"
    )

    print(f"Validation accuracy: {valid_acc:.2%}")
    print(f"Validation Macro F1: {valid_f1:.4f}")

    predictions = model.predict(X_test)

    test_acc = accuracy_score(test_labels, predictions)

    macro_f1 = f1_score(
        test_labels, predictions,
        average="macro"
    )

    print(confusion_matrix(
        test_labels, predictions, labels=[0, 1, 2]
    ))

    print(classification_report(
        test_labels,
        predictions,
        labels=[0, 1, 2],
        target_names=["Left", "Center", "Right"],
        zero_division=0
    ))

    print(f"Test accuracy: {test_acc:.2%}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    # Save model and vectorizer
    joblib.dump(model, "naive_bayes.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()