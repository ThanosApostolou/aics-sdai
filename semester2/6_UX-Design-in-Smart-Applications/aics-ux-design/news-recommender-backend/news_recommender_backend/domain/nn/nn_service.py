from pathlib import Path
import logging
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import int64, float64
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from news_recommender_backend.domain.entities.entities import User
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from news_recommender_backend.domain.news.news_constants import NewsCategoriesEnum



def plot_curves(epochs, hist, metrics_names, plot_file: Path):
    """Plot a curve of one or more classification metrics vs. epoch."""
    # metrics should be one of the names shown in:
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    for m in metrics_names:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)
    plt.legend()
    plt.savefig(plot_file)
    plt.show(block=False)
    logging.debug("Defined the plot_curve function.")



def get_class_names() -> list[str]:
    return list(NewsCategoriesEnum)


def get_class_name_to_index_dict() -> dict[NewsCategoriesEnum, int]:
    class_name_to_index_dict: dict[NewsCategoriesEnum, int] = {}
    for index, category in enumerate(NewsCategoriesEnum):
        class_name_to_index_dict[category] = index
    return class_name_to_index_dict

def get_index_to_class_name_dict() -> dict[int, NewsCategoriesEnum]:
    index_to_class_name_dict: dict[int, NewsCategoriesEnum] = {}
    for index, category in enumerate(NewsCategoriesEnum):
        index_to_class_name_dict[index] = category
    return index_to_class_name_dict


def create_metrics():
    metrics = [tf.keras.metrics.MeanAbsoluteError(name='mae'), tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.CategoricalCrossentropy(name='categorical_crossentropy'),  tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    metrics_names = list(map(lambda metric: metric.name, metrics))
    return metrics, metrics_names

def create_model() -> tf.keras.Sequential:
    metrics, _ = create_metrics()
    learning_rate: float = 0.01
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(5,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(NewsCategoriesEnum), activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=metrics)
    return model


def user_to_feature_list(user: User) -> list[float]:
    assert user.questionnaire is not None
    return [1 if user.questionnaire.is_extrovert else 0, user.questionnaire.age, user.questionnaire.educational_level, user.questionnaire.sports_interest, user.questionnaire.arts_interest]

def user_to_label_one_hot_vector(user: User) -> NDArray[float64]:
    assert user.questionnaire is not None
    one_hot_vector = np.zeros(len(NewsCategoriesEnum))
    for index, category in enumerate(NewsCategoriesEnum):
        logging.info(f"index,category={index},{category}")
        if category == user.questionnaire.favorite_category:
            one_hot_vector[index] = 1
    return one_hot_vector


def create_train_test_sets(users: list[User]) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    logging.debug("start create_train_sets")
    X_list = list(map(user_to_feature_list, users))
    X = np.array(X_list, dtype=float64)
    X = preprocessing.normalize(X, axis=0)
    Y_list = list(map(user_to_label_one_hot_vector, users))
    Y = np.array(Y_list, dtype=int64)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
    return X_train, X_test, Y_train, Y_test


def train_model(mymodel: tf.keras.Sequential, metrics_names: list[str], epochs: int, batch_size: int, train_features: NDArray, train_labels: NDArray, plot_file: Path):
    history = mymodel.fit(train_features, train_labels,
                          epochs=epochs, batch_size=batch_size)

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # To track the progression of training, gather a snapshot
    # of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)
    plot_curves(epochs, hist, metrics_names, plot_file)

def evaluate_model(mymodel: tf.keras.Sequential, batch_size: int, test_features: NDArray, test_labels: NDArray):
    evaluation = mymodel.evaluate(test_features, test_labels, batch_size=batch_size)
    logging.info('evaluation')
    return evaluation

def train_evaluate_model(mymodel: tf.keras.Sequential, users: list[User], plot_file: Path) -> tf.keras.Sequential:
    epochs = 80
    batch_size = 10
    _, metrics_names = create_metrics()
    print(f'len(NewsCategoriesEnum)={len(NewsCategoriesEnum)}')
    X_train, X_test, Y_train, Y_test = create_train_test_sets(users)
    logging.info(f"X_train, X_test, y_train, y_test={X_train, X_test, Y_train, Y_test}")
    logging.info(f"Y_train.shape={Y_train.shape}")
    train_model(mymodel, metrics_names, epochs, batch_size, X_train, Y_train, plot_file)
    evaluation = evaluate_model(mymodel, batch_size, X_test, Y_test)
    logging.info(f"metrics_names=\n{metrics_names}")
    logging.info(f"evaluation=\n{evaluation}")
    return mymodel


def predict_model(mymodel: tf.keras.Sequential, batch_size: int, features: list[list[float]]):
    logging.debug("start predict_model")
    # probability_model = tf.keras.Sequential([mymodel, tf.keras.layers.Softmax()])
    predictions = mymodel.predict(features, batch_size=batch_size)
    logging.info(f"predictions={predictions}")
    logging.debug("end predict_model")
    return predictions

def predict_model_for_user(mymodel: tf.keras.Sequential, user: User) -> NewsCategoriesEnum:
    logging.debug("start predict_model")
    batch_size: int = 10
    user_features = user_to_feature_list(user)
    predictions = predict_model(mymodel, batch_size, [user_features])
    predicted_classes = np.argmax(predictions, axis = 1)
    predicted_class = predicted_classes[0]
    predicted_category: NewsCategoriesEnum | None = None
    for index, category in enumerate(NewsCategoriesEnum):
        if index == predicted_class:
            predicted_category = category

    assert predicted_category is not None
    logging.info(f"predictions={predictions}")
    logging.info(f"predicted_classes={predicted_classes}")
    logging.debug("end predict_model")
    return predicted_category
