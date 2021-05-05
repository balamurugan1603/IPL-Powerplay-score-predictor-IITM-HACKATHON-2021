from tensorflow.keras.models import load_model
from tensorflow import convert_to_tensor, float64
import pandas as pd
import re
import pickle

def predictRuns(testInput):

    # loading models
    combined_model = load_model("Combined_model")
    batting_model = load_model("batting_model")
    elasticNet_model = pickle.load(open("elasticNetModel", "rb"))

    # loading data
    testCase = pd.read_csv(testInput)

    # preprocessing venues
    venue_pp_avg = {
        "edengardens": 46.32,
        "sardarpatelstadiummotera": 47.34,
        "narendramodistadium": 47.34,
        "sharjahcricketstadium": 44.34,
        "dubaiinternationalcricketstadium": 41.82,
        "mchinnaswamystadium": 46.35,
        "machidambaramstadiumchepauk": 44.50,
        "machidambaramstadium": 44.50,
        "machidambaramstadiumchepaukchennai": 44.50,
        "wankhedestadium": 44.17,
        "wankhedestadiummumbai": 44.17,
        "arunjaitleystadium": 47.88,
        "ferozshahkotla": 47.88
    }

    testCase.iloc[0, 0] = re.sub("[^A-Za-z]+", '', testCase.iloc[0, 0].lower())
    testCase = testCase.replace(testCase.iloc[0, 0], venue_pp_avg[testCase.iloc[0, 0]])

    # binary encoding innings
    testCase.iloc[0, 1] = testCase.iloc[:, 1].replace({1: 0, 2: 1})

    # preprocessing batting_team
    team_batting_points = {
        "chennaisuperkings": 44.97,
        "delhicapitals": 46.678,
        "mumbaiindians": 44.32,
        "kolkataknightriders": 42.135,
        "sunrisershyderabad": 45.74,
        "rajasthanroyals": 45.133,
        "royalchallengersbangalore": 49.125,
        "punjabkings": 48.125,
        "kingsxipunjab": 48.125
    }

    testCase.iloc[0, 2] = re.sub("[^A-Za-z]+", '', testCase.iloc[0, 2].lower())
    testCase.iloc[0, 2] = testCase.iloc[:, 2].replace(testCase.iloc[0, 2], team_batting_points[testCase.iloc[0, 2]])

    # preprocessing bowling_team
    team_bowling_points = {
        "chennaisuperkings": 44.83,
        "delhicapitals": 44.77,
        "mumbaiindians": 41.9,
        "kolkataknightriders": 44.9,
        "sunrisershyderabad": 44.08,
        "rajasthanroyals": 45.99,
        "royalchallengersbangalore": 46.78,
        "punjabkings": 49.48,
        "kingsxipunjab": 49.48
    }
    testCase.iloc[0, 3] = re.sub("[^A-Za-z]+", '', testCase.iloc[0, 3].lower())
    testCase.iloc[0, 3] = testCase.iloc[:, 3].replace(testCase.iloc[0, 3], team_bowling_points[testCase.iloc[0, 3]])

    # n_wickets and n_bowlers preprocessing
    testCase.iloc[0, 4] = len(testCase.iloc[0, 4].split(','))-2
    testCase.iloc[0, 5] = len(testCase.iloc[0, 5].split(','))

    # normalising
    venue_max = 47.88
    batting_team_max = 61.2
    bowling_team_max = 62.5
    n_wickets_max = 6
    n_bowlers_max = 6

    testCase["venue"] = testCase["venue"] / venue_max
    testCase["batting_team"] = testCase["batting_team"] / batting_team_max
    testCase["bowling_team"] = testCase["bowling_team"] / bowling_team_max
    testCase["batsmen"] = testCase["batsmen"] / n_wickets_max
    testCase["bowlers"] = testCase["bowlers"] / n_bowlers_max

    # converting data to tensor
    testCaseCombined = convert_to_tensor(testCase.iloc[0:1, 0:6], dtype=float64)
    testCaseBatting = convert_to_tensor(testCase.iloc[0:1, [0, 1, 2, 4]], dtype=float64)

    # predicting
    combined_prediction = combined_model.predict(testCaseCombined)
    batting_prediction = batting_model.predict(testCaseBatting)
    elasticNet_prediction = elasticNet_model.predict(testCaseCombined)

    # computing deviation
    if combined_prediction <= 44:
        deviation = (-1 * combined_prediction) / 47.58
    else:
        deviation = 0

    # computing final result (runs)
    prediction = int((batting_prediction*0.125)+(elasticNet_prediction*0.125)+(combined_prediction*0.75)+deviation)

    return prediction


# replace testInput.csv with actual path
print("Predicted Runs : ", predictRuns("testInput.csv"))
