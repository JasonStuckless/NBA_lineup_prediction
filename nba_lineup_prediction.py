import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# List of matchup file years
train_test_years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]

# Load CSV file into a DataFrame
def load_data(file):
    return pd.read_csv(file)

# Store prediction results
results = []

# Loop for each training/testing years pair
for i in range(len(train_test_years) - 1):
    # Assign training/testing year values
    train_year = train_test_years[i]
    test_year = train_test_years[i + 1]

    # Identify data set file names for training/testing years
    train_file = f"matchups-{train_year}.csv"
    test_file = f"matchups-{test_year}.csv"

    # Load CSV files into panda dataframes
    df_train = load_data(train_file)
    df_test = load_data(test_file)

    # Select only the relevant columns for modeling
    allowed_columns = ["home_team", "away_team", "home_0", "home_1", "home_2", "home_3", "home_4",
                       "away_0", "away_1", "away_2", "away_3", "away_4", "outcome"]
    df_train = df_train[allowed_columns].copy()
    df_test = df_test[allowed_columns].copy()

    # Compute individual player win rates
    player_win_rates = df_train.melt(id_vars=["outcome"], value_vars=["home_0", "home_1", "home_2", "home_3", "home_4",
                                                                      "away_0", "away_1", "away_2", "away_3", "away_4"])
    player_win_rates = player_win_rates.groupby("value")["outcome"].mean().to_dict()

    # Map player win rates to both training and testing datasets
    for col in ["home_0", "home_1", "home_2", "home_3", "home_4", "away_0", "away_1", "away_2", "away_3", "away_4"]:
        df_train[col + "_win_rate"] = df_train[col].map(player_win_rates)
        df_test[col + "_win_rate"] = df_test[col].map(player_win_rates)

    # Compute synergy score using win ratio
    def compute_synergy(df, players):
        synergy = {}
        for _, row in df.iterrows():
            for i in range(len(players)):
                for j in range(i + 1, len(players)):
                    pair = tuple(sorted([row[players[i]], row[players[j]]]))
                    if pair not in synergy:
                        synergy[pair] = {"wins": 0, "games": 0}
                    synergy[pair]["games"] += 1
                    if row["outcome"] == 1:
                        synergy[pair]["wins"] += 1
        synergy_scores = {k: v["wins"] / v["games"] if v["games"] > 0 else 0 for k, v in synergy.items()}
        return synergy_scores

    # Compute synergy scores for the training dataset
    synergy_scores = compute_synergy(df_train,
                                     ["home_0", "home_1", "home_2", "home_3", "home_4", "away_0", "away_1", "away_2",
                                      "away_3", "away_4"])

    # Calculates the average synergy score for all unique player pairs
    def map_synergy_score(row, players):
        scores = []
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                pair = tuple(sorted([row[players[i]], row[players[j]]]))
                scores.append(synergy_scores.get(pair, 0))
        return np.mean(scores) if scores else 0

    # Adds synergy scores to train/test data by averaging player pair synergy.
    df_train["synergy_score"] = df_train.apply(lambda row: map_synergy_score(row, df_train.columns[2:12]), axis=1)
    df_test["synergy_score"] = df_test.apply(lambda row: map_synergy_score(row, df_test.columns[2:12]), axis=1)

    # Encode categorical features
    encoder = LabelEncoder()
    categorical_columns = ["home_team", "away_team", "home_0", "home_1", "home_2", "home_3",
                           "away_0", "away_1", "away_2", "away_3", "away_4"]

    # Apply encoding and handle unknown values in test set
    for col in categorical_columns + ["home_4"]:
        df_train[col] = encoder.fit_transform(df_train[col])
        df_test[col] = df_test[col].apply(lambda x: x if x in encoder.classes_ else "unknown")
        encoder.classes_ = np.append(encoder.classes_, "unknown")
        df_test[col] = encoder.transform(df_test[col])

    # Create parameters to train model
    X_train = df_train.drop(columns=["home_4", "outcome"])
    y_train = df_train["home_4"]
    X_test = df_test.drop(columns=["home_4", "outcome"])
    y_test = df_test["home_4"]

    # Initializes and trains XGBoost model
    model = XGBClassifier(eval_metric='mlogloss', max_depth=10, n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Get predicted probabilities for test data
    y_prob = model.predict_proba(X_test)
    # Extract indices of top 3 predicted classes per row
    top_3_preds = np.argsort(y_prob, axis=1)[:, -3:]
    # Convert indices to actual player class labels for top 3 predictions
    possible_players = np.array(model.classes_)[top_3_preds]

    # Generate and evaluate top 1 predictions
    y_pred_top1 = model.predict(X_test)
    top_1_accuracy = accuracy_score(y_test, y_pred_top1)
    top_1_precision = precision_score(y_test, y_pred_top1, average='weighted', zero_division=0)
    top_1_recall = recall_score(y_test, y_pred_top1, average='weighted', zero_division=0)
    top_1_f1 = f1_score(y_test, y_pred_top1, average='weighted', zero_division=0)

    # Generate and evaluate top 3 predictions
    top_3_accuracy = np.mean([y_test.iloc[i] in possible_players[i] for i in range(len(y_test))])
    top_3_precision = precision_score(y_test, possible_players[:, -1], average='weighted', zero_division=0)
    top_3_recall = recall_score(y_test, possible_players[:, -1], average='weighted', zero_division=0)
    top_3_f1 = f1_score(y_test, possible_players[:, -1], average='weighted', zero_division=0)

    # Store evaluation metrics for current train-test year pair
    results.append((train_year, test_year, top_1_accuracy, top_1_precision, top_1_recall, top_1_f1, top_3_accuracy,
                    top_3_precision, top_3_recall, top_3_f1))

    # Print results
    print(f"Training Year: {train_year} -> Testing Year: {test_year}")
    print(
        f"Top 1 Accuracy: {top_1_accuracy:.4f}, Precision: {top_1_precision:.4f}, Recall: {top_1_recall:.4f}, F1-score: {top_1_f1:.4f}")
    print(
        f"Top 3 Accuracy: {top_3_accuracy:.4f}, Precision: {top_3_precision:.4f}, Recall: {top_3_recall:.4f}, F1-score: {top_3_f1:.4f}")
    print("--------------------------------------------")

# Save results to CSV
results_df = pd.DataFrame(results,
                          columns=["Train Year", "Test Year", "Top 1 Accuracy", "Top 1 Precision", "Top 1 Recall",
                                   "Top 1 F1-score", "Top 3 Accuracy", "Top 3 Precision", "Top 3 Recall",
                                   "Top 3 F1-score"])
results_df.to_csv("training_testing_results.csv", index=False)
print("Results saved to training_testing_results.csv")