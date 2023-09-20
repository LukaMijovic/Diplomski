import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math

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
# X_train, X_test, y_train, y_test

scaler = StandardScaler()

features_train = scaler.fit_transform(features_train)
features_test = scaler.fit_transform(features_test)

svm = SVR(kernel="linear")
svm.fit(features_train, target_train)
preds = svm.predict(features_test)

sns.kdeplot(target_test, label="Actual values", color="blue")
sns.kdeplot(preds, label="Predictions", color="red")
plt.xlim(-10000, 10000)
plt.show()


mae = round(mean_absolute_error(target_test, preds), 3)
rmse = round(math.sqrt(mean_squared_error(target_test, preds)), 3)
r2 = round(r2_score(target_test, preds), 6)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")
