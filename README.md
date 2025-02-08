import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

print(f"Train Shape: {train.shape}, Test Shape: {test.shape}")


num_cols = train.select_dtypes(include=['number']).columns.drop('SalePrice', errors='ignore')
cat_cols = train.select_dtypes(include=['object']).columns


num_imputer = SimpleImputer(strategy="median")
train[num_cols] = num_imputer.fit_transform(train[num_cols])
test[num_cols] = num_imputer.transform(test[num_cols])


cat_imputer = SimpleImputer(strategy="most_frequent")
train[cat_cols] = cat_imputer.fit_transform(train[cat_cols])
test[cat_cols] = cat_imputer.transform(test[cat_cols])


train[cat_cols] = train[cat_cols].astype(str)
test[cat_cols] = test[cat_cols].astype(str)

encoder = LabelEncoder()
for col in cat_cols:
    train[col] = encoder.fit_transform(train[col])
    test[col] = encoder.transform(test[col])


X = train.drop(columns=['Id', 'SalePrice'])
y = train['SalePrice']
X_test = test.drop(columns=['Id'])


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f"Validation RMSE: {rmse}")

predictions = model.predict(X_test)
submission = pd.DataFrame({'Id': test['Id'].astype(int), 'SalePrice': predictions})

submission.to_csv('/kaggle/working/submission.csv', index=False)

