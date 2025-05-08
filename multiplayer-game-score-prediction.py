import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class PlayerMatchData:

    def __init__(self, kills, deaths, assists, damage_dealt, damage_received, healing_done, objective_time, won=None):
        self.kills = kills
        self.deaths = deaths
        self.assists = assists
        self.damage_dealt = damage_dealt
        self.damage_received = damage_received
        self.healing_done = healing_done
        self.objective_time = objective_time
        self.won = won

    def to_dict(self):
        return {
            'kills': self.kills,
            'deaths': self.deaths,
            'assists': self.assists,
            'damage_dealt': self.damage_dealt,
            'damage_received': self.damage_received,
            'healing_done': self.healing_done,
            'objective_time': self.objective_time,
            'won': self.won
        }

class VictoryPredictor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression()

    def train(self, data: DataFrame):

        X = data.drop(['won'], axis=1)
        y = data['won']

        x_scaled = self.scaler.fit_transform(X)

        self.model.fit(x_scaled, y)

    def predict(self, player: PlayerMatchData):

        df = pd.DataFrame([player.to_dict()])
        df = df.drop(['won'], axis=1)

        df_scaled = self.scaler.transform(df)

        return int(self.model.predict(df_scaled)[0])

def generate_synthetic_data():

    iteration = range(1, 50)
    player_matchs_data = []

    for i in iteration:
        kills = np.random.poisson(5)
        deaths = np.random.poisson(3)
        assists = np.random.poisson(2)
        damage_dealt = kills * 300 + np.random.normal(0, 100)
        damage_received = deaths * 400 + np.random.normal(0, 100)
        healing_done = np.random.randint(0, 300)
        objective_time = np.random.randint(0, 120)

        if damage_dealt > damage_received and kills > deaths:
            won = 1
        else:
            won = 0

        player_match = PlayerMatchData(kills, deaths, assists, damage_dealt, damage_received, healing_done, objective_time, won)
        player_matchs_data.append(player_match)

    data = pd.DataFrame([player_match.to_dict() for player_match in player_matchs_data])

    return data

victoryPredictor = VictoryPredictor()
victoryPredictor.train(generate_synthetic_data())

won = victoryPredictor.predict(PlayerMatchData(30,  6, 15, 4000, 400, 50, 30))

if won == 1:
    print("The player has won the match")
else:
    print("The player has lost the match")
