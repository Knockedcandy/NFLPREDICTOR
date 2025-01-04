import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random

# File paths
path = "C:\\Users\\Emili\\NFLPREDICTOR\\NFLDATA"
file_paths = {
    "data": f"{path}\\data.csv",
    "differential": f"{path}\\differential.csv",
    "teams": f"{path}\\teams.csv",
    "abbreviations": f"{path}\\abbreviations.csv"
}

# Load datasets
data_df = pd.read_csv(file_paths["data"])
differential_df = pd.read_csv(file_paths["differential"])

# Preprocessing
# Process `data_df`
data_df['spread_favorite'] = pd.to_numeric(data_df['spread_favorite'], errors='coerce')
data_features = data_df[['spread_favorite', 'over_under_line', 'team_home', 'team_away']].copy()

# Drop rows with missing values in `data_features`
data_features.dropna(inplace=True)

# Encode categorical columns (team names)
all_teams = pd.concat([data_features['team_home'], data_features['team_away']]).unique()
team_encoding = {team: idx for idx, team in enumerate(all_teams)}
data_features['team_home'] = data_features['team_home'].map(team_encoding)
data_features['team_away'] = data_features['team_away'].map(team_encoding)

# Process `differential_df`
differential_features = differential_df[['team_home', 'team_away', 'score_home', 'score_away']].copy()

differential_features.dropna(inplace=True)

differential_features['team_home'] = differential_features['team_home'].map(team_encoding)
differential_features['team_away'] = differential_features['team_away'].map(team_encoding)


differential_features['home_win'] = (differential_features['score_home'] > differential_features['score_away']).astype(int)

# Combine features
model_features = pd.concat([data_features, differential_features[['score_home', 'score_away', 'home_win']]], axis=1)

# Drop rows with missing values in `model_features`
model_features.dropna(inplace=True)

# Select feature columns and target
feature_columns = ['spread_favorite', 'over_under_line', 'team_home', 'team_away', 'score_home', 'score_away']
X = model_features[feature_columns]
y = model_features['home_win']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict team records for the next 5 years
def predict_records_for_years(teams, years=5):
    records = {team: {"Wins": 0, "Losses": 0, "Games": 0} for team in teams}

    for year in range(2024, 2024 + years):
        print(f"Year: {year}")

        # Reset records for each team at the beginning of the year
        for team in teams:
            records[team] = {"Wins": 0, "Losses": 0, "Games": 0}

        # Simulate games for the year
        for _ in range(256):  #Makes sure each team plays 17 games each
            # Randomly choose a home and away team
            home_team = random.choice(teams)
            away_team = random.choice([team for team in teams if team != home_team])

            # Ensure no team plays more than 17 games
            while records[home_team]["Games"] >= 17 or records[away_team]["Games"] >= 17:
                home_team = random.choice(teams)
                away_team = random.choice([team for team in teams if team != home_team])

            # Simulate input features
            input_features = {
                "spread_favorite": random.uniform(-10, 10),
                "over_under_line": random.uniform(30, 60),
                "team_home": team_encoding.get(home_team, -1),
                "team_away": team_encoding.get(away_team, -1),
                "score_home": 0,  
                "score_away": 0   
            }
            input_df = pd.DataFrame([input_features])

           
            input_df = input_df[feature_columns]

            # Predict outcome
            home_win = model.predict(input_df)[0]

            # Update records
            if home_win:
                records[home_team]["Wins"] += 1
                records[away_team]["Losses"] += 1
            else:
                records[away_team]["Wins"] += 1
                records[home_team]["Losses"] += 1

            
            records[home_team]["Games"] += 1
            records[away_team]["Games"] += 1

        # Print results for the year
        for team, record in records.items():
            print(f"  {team}: {record['Wins']} Wins, {record['Losses']} Losses")

# Predict records
teams = list(team_encoding.keys())
predict_records_for_years(teams)
