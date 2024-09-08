import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# Load data
df = pd.read_pickle("/home/avinash/Downloads/data-science-template-main/data/interim/01_data_processed.pkl")

# Create a training and test set
df_train = df.drop(["participant", "category", "set"], axis=1)
X = df_train.drop("label", axis=1)
y = df_train["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Plot label distribution
fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(kind="bar", ax=ax, color="lightblue", label="Total")
y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()

# Split feature subsets
basic_features = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
square_features = ["acc_r", "gyr_r"]
pca_features = ["pca_1", "pca_2", "pca3"]
time_features = [f for f in df_train.columns if "_temp_" in f]
frequency_features = [f for f in df_train.columns if "_freq_" in f or "_pse" in f]
cluster_features = ["cluster"]

feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + frequency_features + cluster_features))

# Perform forward feature selection using simple decision tree
learner = ClassificationAlgorithms()
max_features = 10
selected_features, ordered_features, ordered_scores = learner.forward_selection(max_features, X_train, y_train)

# Plot forward selection results
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, max_features + 1, 1))
plt.show()

# Grid search for best hyperparameters and model selection
possible_feature_sets = [feature_set_1, feature_set_2, feature_set_3, feature_set_4, selected_features]
feature_names = ["feature_set_1", "feature_set_2", "feature_set_3", "feature_set_4", "Selected_features"]
iterations = 1
score_df = pd.DataFrame()

for i, f in zip(range(len(possible_feature_sets)), feature_names):
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    performance_test_nn = 0
    performance_test_rf = 0

    for _ in range(iterations):
        # Training neural network
        class_train_y, class_test_y, _, _ = learner.feedforward_neural_network(
            selected_train_X, y_train, selected_test_X, gridsearch=False)
        performance_test_nn += accuracy_score(y_test, class_test_y)

        # Training random forest
        class_train_y, class_test_y, _, _ = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True)
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn /= iterations
    performance_test_rf /= iterations

    # Training KNN
    class_train_y, class_test_y, _, _ = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True)
    performance_test_knn = accuracy_score(y_test, class_test_y)

    # Training decision tree
    class_train_y, class_test_y, _, _ = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True)
    performance_test_dt = accuracy_score(y_test, class_test_y)

    # Training naive bayes
    class_train_y, class_test_y, _, _ = learner.naive_bayes(
        selected_train_X, y_train, selected_test_X)
    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame({
        "model": models,
        "feature_set": f,
        "accuracy": [
            performance_test_nn,
            performance_test_rf,
            performance_test_knn,
            performance_test_dt,
            performance_test_nb,
        ],
    })
    score_df = pd.concat([score_df, new_scores])

# Create a grouped bar plot to compare the results
score_df = score_df.sort_values(by="accuracy", ascending=False)
plt.figure(figsize=(10, 10))
sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)  # Accuracy is between 0 and 1
plt.legend(loc="lower right")
plt.show()

# Select best model and evaluate results
class_train_y, class_test_y, _, class_test_prob_y = learner.random_forest(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True)
accuracy = accuracy_score(y_test, class_test_y)
classes = y_test.unique()
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# Create confusion matrix plot
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Select train and test data based on participant
participant_df = df.drop(["set", "category"], axis=1)
train_mask = participant_df["participant"] != "A"
X_train = participant_df[train_mask].drop(["label", "participant"], axis=1)
y_train = participant_df[train_mask]["label"]
X_test = participant_df[~train_mask].drop(["label", "participant"], axis=1)
y_test = participant_df[~train_mask]["label"]

# Plot label distribution for participant-based split
fig, ax = plt.subplots(figsize=(10, 5))
y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()

# Use best model again and evaluate results
class_train_y, class_test_y, _, class_test_prob_y = learner.random_forest(
    X_train[selected_features], y_train, X_test[selected_features], gridsearch=True)
accuracy = accuracy_score(y_test, class_test_y)
classes = y_test.unique()
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# Create confusion matrix plot
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
