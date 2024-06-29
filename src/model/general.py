from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, \
    KBinsDiscretizer, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.dates import num2date


# def func2(x):
#     return np.arctan(x) * x + x


# def func4(x):
#     return np.arctan(x) * x - x * np.sin(x)


# def func5(x):
#     return np.arctan(x) * x


# def func6(x):
#     return x * np.arctan(x) + x * np.cos(x)


# Scalers = [MinMaxScaler(), StandardScaler(), RobustScaler(), FunctionTransformer(np.sin),
#            FunctionTransformer(func2), FunctionTransformer(func4), FunctionTransformer(func5),
#            FunctionTransformer(func6)]


class Prediction:
    def __init__(self, figi, folder):
        self.figi = figi
        self.folder = folder
        self.bids = 1
        self.asks = 1
        self.w_bid = 0
        self.w_ask = 0
        self.buys = 0
        self.sells = 0
        self.price = 0
        self.main_model = joblib.load(os.path.join(f'{folder}/' + self.figi, 'ModelMain.pkl'))
        self.bid_model = joblib.load(os.path.join(f'{folder}/' + self.figi, 'ModelBid.pkl'))
        self.ask_model = joblib.load(os.path.join(f'{folder}/' + self.figi, 'ModelAsk.pkl'))
        self.logic_model = joblib.load(os.path.join(f'{folder}/' + self.figi, 'ModelLogic.pkl'))
        self.poly21 = joblib.load(os.path.join(f'{folder}/' + self.figi, 'poly21.joblib'))
        self.poly22 = joblib.load(os.path.join(f'{folder}/' + self.figi, 'poly22.joblib'))
        self.poly1 = joblib.load(os.path.join(f'{folder}/' + self.figi, 'poly1.joblib'))
        self.predictions = []
        self.prices = []

    def predict(self):
        directory = f'{self.folder}/{self.figi}'
        if self.buys != 0:
            if self.sells != 0:
                data1 = self.buys / self.sells
            else:
                data1 = self.buys
        else:
            data1 = 1 / self.sells
        data2 = self.w_bid
        data3 = self.w_ask
        data4 = self.bids
        data5 = self.asks
        data7 = data4 / data5
        pointbid = self.poly21.transform(np.array([[data1, data3, data4, data5]]))
        pointask = self.poly22.transform(np.array([[data1, data2, data4, data5]]))

        predicted_bid = self.bid_model.predict(pointbid)[0]
        predicted_ask = self.ask_model.predict(pointask)[0]
        predicted_price = \
        self.main_model.predict(self.poly1.transform(np.array([[predicted_bid, predicted_ask, data1, data7]])))[0]
        self.prices.append(self.price)
        self.predictions.append(predicted_price)

        # Check the accuracy of the model
        # функция логики, определяющая ложное ли срабатывание или нет


def train(figi, folder):
    directory = f'{folder}/{figi}'
    get_op = GetData(figi, folder)
    x1, y1, x2, y2, x3, y3 = GetData.getdata1234(get_op)
    x5, y5, x6, y6 = GetData.getdata567(get_op)
    Data1, Data2, Data3, Data4, Data5, Data6, Data7 = GetData.rearrangement(get_op, x1, y1, x2, y2, x3, y3, x5, y5, x6,
                                                                            y6)
    model_main = LinearRegression()
    Data8, Data9 = GetData.getdata89(get_op, len(Data5))
    Data10 = Data8 / Data9
    Data11 = np.array(Data5) / np.array(Data6)
    model_bid = LinearRegression()
    model_ask = LinearRegression()
    X21 = np.array([Data10, Data3, Data5, Data6]).T
    X22 = np.array([Data10, Data2, Data5, Data6]).T
    y3 = np.array(Data3)
    y2 = np.array(Data2)
    X1 = np.array([Data2, Data3, Data10, Data11]).T
    y1 = np.array(Data1)
    degrees = [1]
    best_error = 10000000000000
    best_degree = 0
    for degree in degrees:
        poly22 = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly22 = poly22.fit_transform(X22)
        model_ask.fit(X_poly22, y3)
        y3_pred = model_ask.predict(X_poly22)
        error_ask = mean_squared_error(y3, y3_pred)
        if error_ask < best_error:
            best_error = error_ask
            best_degree = degree
    print(best_degree)
    poly22 = PolynomialFeatures(degree=best_degree, include_bias=True)
    X_poly22 = poly22.fit_transform(X22)
    model_ask.fit(X_poly22, y3)
    best_error = 10000000000000
    best_degree = 0
    for degree in degrees:
        poly21 = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly21 = poly21.fit_transform(X21)
        model_bid.fit(X_poly21, y2)
        y2_pred = model_bid.predict(X_poly21)
        error_bid = mean_squared_error(y2, y2_pred)
        if error_bid < best_error:
            best_error = error_bid
            best_degree = degree
    print(best_degree)
    poly21 = PolynomialFeatures(degree=best_degree, include_bias=True)
    X_poly21 = poly21.fit_transform(X21)
    model_bid.fit(X_poly21, y2)
    best_error = 10000000000000
    best_degree = 0
    for degree in degrees:
        poly1 = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly1 = poly1.fit_transform(X1)
        model_main.fit(X_poly1, y1)
        y1_pred = model_main.predict(X_poly1)
        error_main = mean_squared_error(y1, y1_pred)
        if error_main < best_error:
            best_error = error_main
            best_degree = degree
    print(best_degree)
    poly1 = PolynomialFeatures(degree=best_degree, include_bias=True)
    X_poly1 = poly1.fit_transform(X1)
    model_main.fit(X_poly1, y1)
    joblib.dump(poly22, os.path.join(directory, 'poly22.joblib'))
    joblib.dump(poly21, os.path.join(directory, 'poly21.joblib'))
    joblib.dump(poly1, os.path.join(directory, 'poly1.joblib'))
    joblib.dump(model_main, os.path.join(directory, 'ModelMain.pkl'))
    joblib.dump(model_bid, os.path.join(directory, 'ModelBid.pkl'))
    joblib.dump(model_ask, os.path.join(directory, 'ModelAsk.pkl'))


def train_logic(figi, folder):
    directory = f'{folder}/{figi}'
    get_op = GetData(figi, folder)
    poly21 = joblib.load(os.path.join(directory, 'poly21.joblib'))
    poly22 = joblib.load(os.path.join(directory, 'poly22.joblib'))
    poly1 = joblib.load(os.path.join(directory, 'poly1.joblib'))
    model_main = joblib.load(os.path.join(directory, 'ModelMain.pkl'))
    model_bid = joblib.load(os.path.join(directory, 'ModelBid.pkl'))
    model_ask = joblib.load(os.path.join(directory, 'ModelAsk.pkl'))
    x1, y1, x2, y2, x3, y3 = GetData.getdata1234(get_op)
    x5, y5, x6, y6 = GetData.getdata567(get_op)
    Data1, Data2, Data3, Data4, Data5, Data6, Data7 = GetData.rearrangement(get_op, x1, y1, x2, y2, x3, y3, x5, y5, x6,
                                                                            y6)
    Data8, Data9 = GetData.getdata89(get_op, len(Data5))
    Data10 = Data8 / Data9
    Data11 = np.array(Data5) / np.array(Data6)
    Data1 = np.array(Data1)
    ExtrapolatedValues = []
    ExtrapolatedTimes = []
    ExtrapolatedValues1 = []
    ExtrapolatedValues2 = []
    ExtrapolatedTimes1 = []
    Coefficients = []
    logic_samples = []
    for i in range(len(Data2)):
        point2 = np.array([[Data10[i], Data3[i], Data5[i], Data6[i]]])
        point3 = np.array([[Data10[i], Data2[i], Data5[i], Data6[i]]])
        ExtrapolatedValues1.append(model_bid.predict(poly21.transform(point2))[0])
        ExtrapolatedValues2.append(model_ask.predict(poly22.transform(point3))[0])
        ExtrapolatedTimes.append(Data7[i])
    for i in range(len(Data2)):
        point1 = np.array([[ExtrapolatedValues1[i], ExtrapolatedValues2[i], Data10[i], Data11[i]]])
        ExtrapolatedValues.append(model_main.predict(poly1.transform(point1))[0])
        ExtrapolatedTimes1.append(Data4[i])

    for i in range(len((ExtrapolatedValues)) - 15):
        direction = Data1[i] - ExtrapolatedValues[i]
        if direction >= 0:
            time = Data4[i:i + 15]
            price = Data1[i:i + 15]
            list = zip(time, price)
            df = pd.DataFrame(list, columns=['date', 'value'])
            coefficient = stats.linregress(df['date'], df['value'])[0]
            if -0.0000000001 < coefficient < 0.0000000001:
                coefficient = 0
            if coefficient > 100000:
                coefficient = 100000
            Coefficients.append(coefficient)
            if (100 * (Data1[i + 15] - Data1[i]) / Data1[i]) > 0.05:
                logic_samples.append(-1)
            else:
                logic_samples.append(1)
        if direction < 0:
            time = Data4[i:i + 15]
            price = Data1[i:i + 15]
            list = zip(time, price)
            df = pd.DataFrame(list, columns=['date', 'value'])
            coefficient = stats.linregress(df['date'], df['value'])[0]
            if -0.0000000001 < coefficient < 0.0000000001:
                coefficient = 0
            if coefficient > 100000:
                coefficient = 100000
            Coefficients.append(coefficient)
            if (100 * (Data1[i + 15] - Data1[i]) / Data1[i]) > 0.05:
                logic_samples.append(1)
            else:
                logic_samples.append(-1)
    ExtrapolatedValues = np.array(ExtrapolatedValues)
    logic_features = np.array(
        [Data1[:-15], Data2[:-15], Data3[:-15], Data5[:-15], Data6[:-15], ExtrapolatedValues[:-15],
         np.array(Coefficients)])
    features = logic_features.T
    best_accuracy = 0
    best_scaler = None
    for scaler in Scalers:
        scaled_features = scaler.fit_transform(features)
        logic = np.array(logic_samples)
        scaled_features = np.array(scaled_features)
        model_logic = LogisticRegression(class_weight='balanced')
        model_logic.fit(scaled_features, logic)
        accuracy = model_logic.score(scaled_features, logic)
        print(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_scaler = scaler

    joblib.dump(best_scaler, os.path.join(directory, 'scaler.joblib'))
    features = best_scaler.transform(features)
    logic = np.array(logic_samples)
    features = np.array(features)
    model_logic = LogisticRegression(class_weight='balanced')
    model_logic.fit(features, logic)
    joblib.dump(model_logic, os.path.join(directory, 'ModelLogic.pkl'))


def test(figi, folder, risk = 0.8):
    get_op = GetData(figi, 'StocksLogTest')
    x1, y1, x2, y2, x3, y3 = GetData.getdata1234(get_op)
    x5, y5, x6, y6 = GetData.getdata567(get_op)
    Data1, Data2, Data3, Data4, Data5, Data6, Data7 = GetData.rearrangement(get_op, x1, y1, x2, y2, x3, y3, x5, y5, x6,
                                                                            y6)
    Data8, Data9 = GetData.getdata89(get_op, len(Data5))
    Data10 = Data8 / Data9
    Data11 = np.array(Data5) / np.array(Data6)
    directory = f'{folder}/{figi}'
    poly21 = joblib.load(os.path.join(directory, 'poly21.joblib'))
    poly22 = joblib.load(os.path.join(directory, 'poly22.joblib'))
    poly1 = joblib.load(os.path.join(directory, 'poly1.joblib'))
    model_main = joblib.load(os.path.join(directory, 'ModelMain.pkl'))
    model_bid = joblib.load(os.path.join(directory, 'ModelBid.pkl'))
    model_ask = joblib.load(os.path.join(directory, 'ModelAsk.pkl'))
    model_logic = joblib.load(os.path.join(directory, 'ModelLogic.pkl'))
    scaler = joblib.load(os.path.join(directory, 'scaler.joblib'))
    ExtrapolatedValues = []
    ExtrapolatedTimes = []
    ExtrapolatedValues1 = []
    ExtrapolatedValues2 = []
    ExtrapolatedTimes1 = []
    Coefficients = []
    for i in range(len(Data2)):
        point2 = np.array([[Data10[i], Data3[i], Data5[i], Data6[i]]])
        point3 = np.array([[Data10[i], Data2[i], Data5[i], Data6[i]]])
        ExtrapolatedValues1.append(model_bid.predict(poly21.transform(point2))[0])
        ExtrapolatedValues2.append(model_ask.predict(poly22.transform(point3))[0])
        ExtrapolatedTimes.append(Data7[i])
    for i in range(len(Data2)):
        point1 = np.array([[ExtrapolatedValues1[i], ExtrapolatedValues2[i], Data10[i], Data11[i]]])
        ExtrapolatedValues.append(model_main.predict(poly1.transform(point1))[0])
        ExtrapolatedTimes1.append(Data4[i])

    for i in range(15, len(Data2)):
        time = Data4[i - 15:i]
        price = Data1[i - 15:i]
        list = zip(time, price)
        df = pd.DataFrame(list, columns=['date', 'value'])
        coefficient = stats.linregress(df['date'], df['value'])[0]
        Coefficients.append(coefficient)
    profit = 0
    number = 0
    buys = []
    backpack_price = 0
    for i in range(15, len(Data2)):
        point4 = np.array(
            [[Data1[i], Data2[i], Data3[i], Data5[i], Data6[i], ExtrapolatedValues[i], Coefficients[i - 15]]])
        buy_predict = model_logic.predict_proba(scaler.transform(point4))[0][0]
        sell_predict = model_logic.predict_proba(scaler.transform(point4))[0][1]
        best = max(buy_predict, sell_predict)
        if best == buy_predict and risk < best:
            if number < 8:
                buys.append(Data1[i]*1.00045)
                backpack_price = np.mean(buys)
                number += 1
            ax.plot(Data4[i], Data1[i], 'o', color='g')
        if best == sell_predict and risk < best:
            if number - 1 != -1:
                buys = []
                profit += (Data1[i]*0.99965 - backpack_price)*number
                number = 0
                backpack_price = 0
                ax.plot(Data4[i], Data1[i], 'o', color='r')

        if 100*(Data1[i]-Data2[i])/Data2[i] < 0.04:
            print(Data1[i], Data2[i])
            ax.plot(Data4[i], Data1[i], 'o', color='b')

    last = (max(Data1[-50:])*0.99965 - backpack_price)*number
    print(profit, number)


    ax.plot(Data4, Data1, label='Actual')
    plt.ylim(min(Data1), max(Data1))
    ax.plot(ExtrapolatedTimes1, ExtrapolatedValues, label='Predicted')
    ax.legend()
    mistake = 0
    for j in range(len(Data2)):
        mistake += (Data1[j] - ExtrapolatedValues[j]) ** 2
    plt.show()


def main3(figi):
    train(figi = figi, folder = 'StocksLog')
    train_logic(figi = figi, folder = 'StocksLog')
    test(figi = figi, folder = 'StocksLog')
main3('BBG00178PGX3')
#main3('BBG006L8G4H1')

# predict = Prediction('BBG006L8G4H1')
# predict.bids = 1000
# predict.asks = 150
# predict.price = 1234
# predict.buys = 1214
# predict.sells = 1643
# predict.w_bid = 1129
# predict.w_ask = 1400
# Prediction.predict(predict)


# predict = Prediction('BBG006L8G4H1')
# predict.bids = 1000
# predict.asks = 150
# predict.price = 1234
# predict.buys = 1214
# predict.sells = 1643
# predict.w_bid = 1129
# predict.w_ask = 1400
# Prediction.predict(predict)
