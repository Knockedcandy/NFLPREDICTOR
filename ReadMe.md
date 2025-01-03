# NFL Team Record Prediction

This project utilizes a machine learning model to predict NFL team records for the next 5 years based on historical data. The model is built using a RandomForestClassifier and trained on data about team matchups, spread favorites, over/under lines, and team scores.

## Project Overview

This script performs the following tasks:
1. Loads and preprocesses data from CSV files.
2. Encodes team names into numeric values.
3. Trains a Random Forest classifier to predict game outcomes.
4. Simulates NFL seasons to predict team records for the next 5 years based on the trained model.


Output Example (5 year period):


Year: 2024
  BAL: 7 Wins, 7 Losses
  BUF: 6 Wins, 11 Losses
  CIN: 8 Wins, 9 Losses
  CLE: 10 Wins, 7 Losses
  DAL: 7 Wins, 10 Losses
  GB: 7 Wins, 8 Losses
  JAX: 8 Wins, 7 Losses
  KC: 7 Wins, 10 Losses
  MIN: 10 Wins, 3 Losses
  NYJ: 7 Wins, 10 Losses
  PHI: 7 Wins, 8 Losses
  LAC: 7 Wins, 10 Losses
  SF: 8 Wins, 3 Losses
  TEN: 9 Wins, 8 Losses
  DEN: 8 Wins, 9 Losses
  ARI: 8 Wins, 9 Losses
  ATL: 11 Wins, 6 Losses
  IND: 10 Wins, 7 Losses
  MIA: 4 Wins, 6 Losses
  NE: 7 Wins, 10 Losses
  SEA: 7 Wins, 10 Losses
  CAR: 10 Wins, 7 Losses
  NYG: 11 Wins, 6 Losses
  OAK: 5 Wins, 12 Losses
  LAR: 9 Wins, 8 Losses
  WAS: 13 Wins, 4 Losses
  NO: 6 Wins, 5 Losses
  PIT: 8 Wins, 9 Losses
  TB: 10 Wins, 7 Losses
  DET: 7 Wins, 10 Losses
  HOU: 7 Wins, 10 Losses
  CHI: 7 Wins, 10 Losses

  
Year: 2025
  BAL: 10 Wins, 7 Losses
  BUF: 6 Wins, 11 Losses
  CIN: 9 Wins, 8 Losses
  CLE: 9 Wins, 8 Losses
  DAL: 8 Wins, 9 Losses
  GB: 7 Wins, 9 Losses
  JAX: 6 Wins, 11 Losses
  KC: 8 Wins, 4 Losses
  MIN: 6 Wins, 10 Losses
  NYJ: 7 Wins, 10 Losses
  PHI: 15 Wins, 2 Losses
  LAC: 9 Wins, 8 Losses
  SF: 6 Wins, 11 Losses
  TEN: 6 Wins, 11 Losses
  DEN: 10 Wins, 7 Losses
  ARI: 8 Wins, 9 Losses
  ATL: 7 Wins, 7 Losses
  IND: 6 Wins, 7 Losses
  MIA: 5 Wins, 12 Losses
  NE: 7 Wins, 10 Losses
  SEA: 10 Wins, 7 Losses
  CAR: 8 Wins, 9 Losses
  NYG: 8 Wins, 9 Losses
  OAK: 6 Wins, 11 Losses
  LAR: 10 Wins, 7 Losses
  WAS: 6 Wins, 5 Losses
  NO: 11 Wins, 4 Losses
  PIT: 10 Wins, 5 Losses
  TB: 9 Wins, 8 Losses
  DET: 10 Wins, 5 Losses
  HOU: 6 Wins, 5 Losses
  CHI: 7 Wins, 10 Losses

  
Year: 2026
  BAL: 9 Wins, 5 Losses
  BUF: 7 Wins, 10 Losses
  CIN: 8 Wins, 9 Losses
  CLE: 5 Wins, 8 Losses
  DAL: 12 Wins, 5 Losses
  GB: 12 Wins, 5 Losses
  JAX: 9 Wins, 8 Losses
  KC: 9 Wins, 8 Losses
  MIN: 8 Wins, 9 Losses
  NYJ: 7 Wins, 10 Losses
  PHI: 10 Wins, 7 Losses
  LAC: 10 Wins, 5 Losses
  SF: 7 Wins, 8 Losses
  TEN: 7 Wins, 8 Losses
  DEN: 7 Wins, 9 Losses
  ARI: 7 Wins, 9 Losses
  ATL: 10 Wins, 7 Losses
  IND: 10 Wins, 7 Losses
  MIA: 5 Wins, 11 Losses
  NE: 8 Wins, 9 Losses
  SEA: 6 Wins, 11 Losses
  CAR: 7 Wins, 9 Losses
  NYG: 6 Wins, 11 Losses
  OAK: 5 Wins, 12 Losses
  LAR: 9 Wins, 8 Losses
  WAS: 5 Wins, 10 Losses
  NO: 4 Wins, 5 Losses
  PIT: 7 Wins, 10 Losses
  TB: 6 Wins, 6 Losses
  DET: 12 Wins, 5 Losses
  HOU: 10 Wins, 7 Losses
  CHI: 12 Wins, 5 Losses

  
Year: 2027
  BAL: 9 Wins, 7 Losses
  BUF: 9 Wins, 8 Losses
  CIN: 8 Wins, 7 Losses
  CLE: 9 Wins, 8 Losses
  DAL: 7 Wins, 10 Losses
  GB: 7 Wins, 10 Losses
  JAX: 10 Wins, 4 Losses
  KC: 7 Wins, 7 Losses
  MIN: 11 Wins, 6 Losses
  NYJ: 6 Wins, 11 Losses
  PHI: 7 Wins, 10 Losses
  LAC: 8 Wins, 9 Losses
  SF: 6 Wins, 11 Losses
  TEN: 10 Wins, 7 Losses
  DEN: 3 Wins, 9 Losses
  ARI: 9 Wins, 8 Losses
  ATL: 5 Wins, 8 Losses
  IND: 4 Wins, 11 Losses
  MIA: 6 Wins, 11 Losses
  NE: 6 Wins, 8 Losses
  SEA: 8 Wins, 9 Losses
  CAR: 6 Wins, 11 Losses
  NYG: 8 Wins, 9 Losses
  OAK: 6 Wins, 7 Losses
  LAR: 12 Wins, 5 Losses
  WAS: 9 Wins, 8 Losses
  NO: 12 Wins, 5 Losses
  PIT: 10 Wins, 7 Losses
  TB: 11 Wins, 6 Losses
  DET: 8 Wins, 7 Losses
  HOU: 10 Wins, 7 Losses
  CHI: 9 Wins, 5 Losses

  
Year: 2028
  BAL: 9 Wins, 8 Losses
  BUF: 7 Wins, 10 Losses
  CIN: 9 Wins, 8 Losses
  CLE: 7 Wins, 7 Losses
  DAL: 10 Wins, 7 Losses
  GB: 11 Wins, 6 Losses
  JAX: 8 Wins, 9 Losses
  KC: 7 Wins, 10 Losses
  MIN: 9 Wins, 8 Losses
  NYJ: 9 Wins, 8 Losses
  PHI: 10 Wins, 6 Losses
  LAC: 5 Wins, 7 Losses
  SF: 7 Wins, 7 Losses
  TEN: 8 Wins, 9 Losses
  DEN: 8 Wins, 8 Losses
  ARI: 9 Wins, 6 Losses
  ATL: 9 Wins, 8 Losses
  IND: 6 Wins, 11 Losses
  MIA: 11 Wins, 4 Losses
  NE: 5 Wins, 12 Losses
  SEA: 10 Wins, 7 Losses
  CAR: 3 Wins, 8 Losses
  NYG: 5 Wins, 8 Losses
  OAK: 10 Wins, 5 Losses
  LAR: 9 Wins, 8 Losses
  WAS: 9 Wins, 8 Losses
  NO: 9 Wins, 8 Losses
  PIT: 8 Wins, 9 Losses
  TB: 9 Wins, 7 Losses
  DET: 8 Wins, 8 Losses
  HOU: 3 Wins, 13 Losses
  CHI: 9 Wins, 8 Losses
