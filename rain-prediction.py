import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class WeatherRecord:

    def __init__(self, humidity, atmosphere_pressure, rain=None):
        self.humidity = humidity
        self.atmosphere_pressure = atmosphere_pressure
        self.rain = rain

class WeatherDataGenerator:

    def generate_data(self):

        iteration = range(1, 50)
        weather_data = []

        for i in iteration:
            humidity = np.random.randint(0, 100)
            atmosphere_pressure = np.random.poisson(20)

            if humidity > 40 and atmosphere_pressure > 10:
                rain = 1
            else:
                rain = 0

            weather = WeatherRecord(humidity, atmosphere_pressure, rain)
            weather_data.append(weather)

        data = pd.DataFrame([{"humidity": weather.humidity, "atmosphere_pressure": weather.atmosphere_pressure, "rain": weather.rain } for weather in weather_data])

        print(data)
        return data

class WeatherRainClassifier:

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression()

    def fit(self, data: DataFrame):

        X = data.drop(['rain'], axis=1)
        y = data['rain']

        scaled_X = self.scaler.fit_transform(X)
        self.model.fit(scaled_X, y)

    def predict(self, humidity, pressure):

        data = pd.DataFrame([{'humidity': humidity, 'atmosphere_pressure': pressure}])
        scaled_data = self.scaler.transform(data)
        return int(self.model.predict(scaled_data)[0])


class WeatherPredictionExample:

    def run(self):

        data = WeatherDataGenerator().generate_data()

        weather_rain_classifier = WeatherRainClassifier()
        weather_rain_classifier.fit(data)

        humidity = 80
        pressure = 50

        result = weather_rain_classifier.predict(humidity, pressure)

        if result == 1:
            print(f"With {humidity}% humidity and {pressure} pressure, IS GONNA rain")
        else:
            print(f"With {humidity}% humidity and {pressure}hPa pressure, IS NOT GONNA rain")

WeatherPredictionExample().run()