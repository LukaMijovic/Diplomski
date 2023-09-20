import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
import math
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

data = pd.read_csv("../Data/finalni_dataset_2.csv")
data.drop(columns=["week.deposit.out (t - 3)", "week.deposit.out (t - 2)", "week.deposit.out (t - 1)",
                   "week.deposit.diff (t - 3)", "week.deposit.diff (t - 2)", "week.deposit.diff (t - 1)",
                   "payin.on.prematch (t - 3)", "payin.on.prematch (t - 2)", "payin.on.prematch (t - 1)",
                   "payin.on.live (t - 3)", "payin.on.live (t - 2)", "payin.on.live (t - 1)",
                   "korisnik", "week.in.year"],
          inplace=True)
#data.drop(columns=["korisnik", "week.in.year"], inplace=True)

data_target = data["total.week.diff (t)"]
data_features = data.drop("total.week.diff (t)", axis=1)

features_train, features_test, target_train, target_test = train_test_split(data_features,
                                                                            data_target,
                                                                            test_size=0.3,
                                                                            shuffle=True,
                                                                            random_state=42)

scaler = StandardScaler()

features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.fit_transform(features_test)

mae_list = []
rmse_list = []
r2_list = []
k_list = []

for k in range(7,20):
    print(k)

    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(features_train_scaled, target_train)

    selected_features_train = selector.transform(features_train_scaled)
    selected_features_test = selector.transform(features_test_scaled)

    nnet = Sequential()
    nnet.add(Dense(20, input_dim=selected_features_train.shape[1], activation="relu"))
    nnet.add(Dense(10, activation="relu"))
    nnet.add(Dense(1, activation="linear"))

    nnet.compile(loss="mean_squared_error", optimizer="adam")

    epochs = 100
    batch_size = 64

    nnet.fit(selected_features_train, target_train, epochs=epochs, batch_size=batch_size, verbose=1)

    preds = nnet.predict(selected_features_test)

    mae = round(mean_absolute_error(target_test, preds), 3)
    mae_list.append(mae)

    rmse = round(math.sqrt(mean_squared_error(target_test, preds)), 3)
    rmse_list.append(rmse)

    r2 = round(r2_score(target_test, preds), 6)
    r2_list.append(r2)

    k_list.append(k)


print(f"RMSE list: {rmse_list}")
print(f"MAE list: {mae_list}")
print(f"R2 list: {r2_list}")

min_value_mae = min(mae_list)
min_index_mae = mae_list.index(min_value_mae)

min_value_rmse = min(rmse_list)
min_index_rmse = rmse_list.index(min_value_rmse)

max_value_r2 = max(r2_list)
max_index_r2 = r2_list.index(max_value_r2)

selector = SelectKBest(score_func=mutual_info_regression, k=k_list[min_index_mae])
selector.fit(features_train_scaled, target_train)

selected_features_mask = selector.get_support()
selected_features = features_train.columns[selected_features_mask]
print(f"For MINIMAL value for MAE ({min_value_mae}) best features ({k_list[min_index_mae]}) are: {selected_features}")

selector = SelectKBest(score_func=mutual_info_regression, k=k_list[min_index_rmse])
selector.fit(features_train_scaled, target_train)

selected_features_mask = selector.get_support()
selected_features = features_train.columns[selected_features_mask]
print(f"For MINIMAL value for RMSE ({min_value_rmse}) best features ({k_list[min_index_rmse]}) are: {selected_features}")

selector = SelectKBest(score_func=mutual_info_regression, k=k_list[max_index_r2])
selector.fit(features_train_scaled, target_train)

selected_features_mask = selector.get_support()
selected_features = features_train.columns[selected_features_mask]
print(f"For MAXIMAL value for R2 ({max_value_r2}) best features ({k_list[max_index_r2]}) are: {selected_features}")