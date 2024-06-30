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

# for i in range(15, len(Data2)):
#         time = Data4[i - 15:i]
#         price = Data1[i - 15:i]
#         list = zip(time, price)
#         df = pd.DataFrame(list, columns=['date', 'value'])
#         coefficient = stats.linregress(df['date'], df['value'])[0]
#         Coefficients.append(coefficient)
#     profit = 0
#     number = 0
#     buys = []
#     backpack_price = 0
#     for i in range(15, len(Data2)):
#         point4 = np.array(
#             [[Data1[i], Data2[i], Data3[i], Data5[i], Data6[i], ExtrapolatedValues[i], Coefficients[i - 15]]])
#         buy_predict = model_logic.predict_proba(scaler.transform(point4))[0][0]
#         sell_predict = model_logic.predict_proba(scaler.transform(point4))[0][1]
#         best = max(buy_predict, sell_predict)
#         if best == buy_predict and risk < best:
#             if number < 8:
#                 buys.append(Data1[i]*1.00045)
#                 backpack_price = np.mean(buys)
#                 number += 1
#             ax.plot(Data4[i], Data1[i], 'o', color='g')
#         if best == sell_predict and risk < best:
#             if number - 1 != -1:
#                 buys = []
#                 profit += (Data1[i]*0.99965 - backpack_price)*number
#                 number = 0
#                 backpack_price = 0
#                 ax.plot(Data4[i], Data1[i], 'o', color='r')

#         if 100*(Data1[i]-Data2[i])/Data2[i] < 0.04:
#             print(Data1[i], Data2[i])
#             ax.plot(Data4[i], Data1[i], 'o', color='b')

#     last = (max(Data1[-50:])*0.99965 - backpack_price)*number
#     print(profit, number)


#     ax.plot(Data4, Data1, label='Actual')
#     plt.ylim(min(Data1), max(Data1))
#     ax.plot(ExtrapolatedTimes1, ExtrapolatedValues, label='Predicted')
#     ax.legend()
#     mistake = 0
#     for j in range(len(Data2)):
#         mistake += (Data1[j] - ExtrapolatedValues[j]) ** 2
#     plt.show()


# def main3(figi):
#     train(figi = figi, folder = 'StocksLog')
#     train_logic(figi = figi, folder = 'StocksLog')
#     test(figi = figi, folder = 'StocksLog')
# main3('BBG00178PGX3')
# #main3('BBG006L8G4H1')

# # predict = Prediction('BBG006L8G4H1')
# # predict.bids = 1000
# # predict.asks = 150
# # predict.price = 1234
# # predict.buys = 1214
# # predict.sells = 1643
# # predict.w_bid = 1129
# # predict.w_ask = 1400
# # Prediction.predict(predict)


# # predict = Prediction('BBG006L8G4H1')
# # predict.bids = 1000
# # predict.asks = 150
# # predict.price = 1234
# # predict.buys = 1214
# # predict.sells = 1643
# # predict.w_bid = 1129
# # predict.w_ask = 1400
# # Prediction.predict(predict)
