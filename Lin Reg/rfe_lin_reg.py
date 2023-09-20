from sklearn import feature_selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import math

data = pd.read_csv("./Data/finalni_dataset_2.csv")
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


from sklearn.feature_selection import RFE

rmse_list = []
mae_list = []
r2_score_list = []
k_list = []

# for k in range(6,20):
#     print(k)
#
#     gbr = LinearRegression()
#
#     RFE_selector = RFE(estimator=gbr, n_features_to_select=k, step=1)
#     RFE_selector.fit(features_train, target_train)
#
#     selected_features_train = RFE_selector.transform(features_train)
#     selected_features_test = RFE_selector.transform(features_test)
#
#     gbr.fit(selected_features_train, target_train)
#     RFE_preds = gbr.predict(selected_features_test)
#
#     rmse = round(math.sqrt(mean_squared_error(target_test, RFE_preds)), 3)
#     rmse_list.append(rmse)
#
#     mae = round(mean_absolute_error(target_test, RFE_preds), 3)
#     mae_list.append(mae)
#
#     r2 = round(r2_score(target_test, RFE_preds), 6)
#     r2_score_list.append(r2)
#
#     k_list.append(k)
#
#
# print(f"RMSE list: {rmse_list}")
# print(f"MAE list: {mae_list}")
# print(f"R2 list: {r2_score_list}")
#
# min_value_mae = min(mae_list)
# min_index_mae = mae_list.index(min_value_mae)
#
# min_value_rmse = min(rmse_list)
# min_index_rmse = rmse_list.index(min_value_rmse)
#
# max_value_r2 = max(r2_score_list)
# max_index_r2 = r2_score_list.index(max_value_r2)

gbr = LinearRegression()

# selector = RFE(estimator=gbr, n_features_to_select=k_list[min_index_mae], step=1)
# selector.fit(features_train, target_train)
#
# selected_features_mask = selector.get_support()
# selected_features = features_train.columns[selected_features_mask]
# print(f"For MINIMAL value for MAE ({min_value_mae}) best features ({k_list[min_index_mae]}) are: {selected_features}")
#
# selector = RFE(estimator=gbr, n_features_to_select=k_list[min_index_rmse], step=1)
# selector.fit(features_train, target_train)
#
# selected_features_mask = selector.get_support()
# selected_features = features_train.columns[selected_features_mask]
# print(f"For MINIMAL value for RMSE ({min_value_rmse}) best features ({k_list[min_index_rmse]}) are: {selected_features}")
#
# selector = RFE(estimator=gbr, n_features_to_select=k_list[max_index_r2], step=1)
# selector.fit(features_train, target_train)
#
# selected_features_mask = selector.get_support()
# selected_features = features_train.columns[selected_features_mask]
# print(f"For MAXIMAL value for R2 ({max_value_r2}) best features ({k_list[max_index_r2]}) are: {selected_features}")

index = 13

print("Poceo RFE")
selector = RFE(estimator=gbr, n_features_to_select=index, step=1)
selector.fit(features_train, target_train)

print("Poceo transformaciju")
selected_features_train = selector.transform(features_train)
selected_features_test = selector.transform(features_test)

print("Poceo predvidjanje")
gbr.fit(selected_features_train, target_train)
RFE_preds = gbr.predict(selected_features_test)

sns.kdeplot(target_test, label="Actual values", color="blue")
sns.kdeplot(RFE_preds, label="Predictions", color="red")
plt.xlim(-10000, 10000)
plt.show()

rmse = round(math.sqrt(mean_squared_error(target_test, RFE_preds)), 3)
mae = round(mean_absolute_error(target_test, RFE_preds), 3)
r2 = round(r2_score(target_test, RFE_preds), 6)
print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")

selected_features_mask = selector.get_support()
selected_features = features_train.columns[selected_features_mask]
print(f"Best features {index}) are: {selected_features}")
