import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

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

lin_reg = LinearRegression()
lin_reg.fit(features_train, target_train)
preds = lin_reg.predict(features_test)

predicted = sorted(preds)
actual = pd.concat([target_train, target_test], axis=0)
actual = sorted(target_test)
x = np.arange(len(target_test))

fig, ax = plt.subplots()
# ax.plot(x, actual, label="Actual values", color="blue")
# ax.plot(x, predicted, label="Predicted", color="red")

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

