import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest

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

scaler = StandardScaler()

features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.fit_transform(features_test)


mae_list = []
rmse_list = []
r2_list = []
k_list = []

# for k in range(13,14):
#     print(k)
#
#     RFE_selector = RFE(estimator=svm, n_features_to_select=k, step=1)
#     RFE_selector.fit(features_train_scaled, target_train)
#
#     selected_features_train = RFE_selector.transform(features_train_scaled)
#     selected_features_test = RFE_selector.transform(features_test_scaled)
#
#     svm.fit(selected_features_train, target_train)
#     RFE_preds = svm.predict(selected_features_test)
#
#     rmse = round(math.sqrt(mean_squared_error(target_test, RFE_preds)), 3)
#     rmse_list.append(rmse)
#
#     mae = round(mean_absolute_error(target_test, RFE_preds), 3)
#     mae_list.append(mae)
#
#     r2 = round(r2_score(target_test, RFE_preds), 6)
#     r2_list.append(r2)
#
#     k_list.append(k)
#
#
# print(f"RMSE list: {rmse_list}")
# print(f"MAE list: {mae_list}")
# print(f"R2 list: {r2_list}")

# min_value_mae = min(mae_list)
# min_index_mae = mae_list.index(min_value_mae)
#
# min_value_rmse = min(rmse_list)
# min_index_rmse = rmse_list.index(min_value_rmse)
#
# max_value_r2 = max(r2_list)
# max_index_r2 = r2_list.index(max_value_r2)

# selector = RFE(estimator=svm, n_features_to_select=k_list[min_index_mae], step=1)
# selector.fit(features_train_scaled, target_train)
#
# selected_features_mask = selector.get_support()
# selected_features = features_train.columns[selected_features_mask]
# print(f"For MINIMAL value for MAE ({min_value_mae}) best features ({k_list[min_index_mae]}) are: {selected_features}")

selector_f = SelectKBest(score_func=f_regression, k=20)
selector_f.fit(features_train_scaled, target_train)

selected_features_train_f = selector_f.transform(features_train_scaled)
selected_features_test_f = selector_f.transform(features_test_scaled)

svm = SVR(kernel="linear")
svm.fit(selected_features_train_f, target_train)
preds_f = svm.predict(selected_features_test_f)

rmse_f = round(math.sqrt(mean_squared_error(target_test, preds_f)), 3)
mae_f = round(mean_absolute_error(target_test, preds_f), 3)
r2_f = round(r2_score(target_test, preds_f), 6)

selected_features_mask_f = selector_f.get_support()
selected_features_f = features_train.columns[selected_features_mask_f]
print("Trenutno stanje nakon FISHERA:")
print(f"RMSE: {rmse_f}, MAE: {mae_f}, R2: {r2_f}")
print(f"Features: {selected_features_f}")

data_final = data[selected_features_f]

features_train, features_test, target_train, target_test = train_test_split(data_features,
                                                                            data_target,
                                                                            test_size=0.3,
                                                                            shuffle=True,
                                                                            random_state=42)

features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.fit_transform(features_test)

k = 13

selector = RFE(estimator=svm, n_features_to_select=k, step=1)
selector.fit(features_train_scaled, target_train)

selected_features_train = selector.transform(features_train_scaled)
selected_features_test = selector.transform(features_test_scaled)

svm.fit(selected_features_train, target_train)
RFE_preds = svm.predict(selected_features_test)

rmse = round(math.sqrt(mean_squared_error(target_test, RFE_preds)), 3)
mae = round(mean_absolute_error(target_test, RFE_preds), 3)
r2 = round(r2_score(target_test, RFE_preds), 6)


selected_features_mask = selector.get_support()
selected_features = features_train.columns[selected_features_mask]
#print(f"For MINIMAL value for RMSE ({min_value_rmse}) best features ({k_list[min_index_rmse]}) are: {selected_features}")
print(f"For {k} features:")
print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")
print(f"Features: {selected_features}")

# selector = RFE(estimator=svm, n_features_to_select=k_list[max_index_r2], step=1)
# selector.fit(features_train_scaled, target_train)
#
# selected_features_mask = selector.get_support()
# selected_features = features_train.columns[selected_features_mask]
# print(f"For MAXIMAL value for R2 ({max_value_r2}) best features ({max_index_r2 + 1}) are: {selected_features}")
