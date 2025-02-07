import pandas as pd
import numpy as np

data = pd.read_csv('data/hcvdat0.csv').drop(columns=["Unnamed: 0"], errors="ignore")
print(data.head())

data['AgeCat'] = pd.cut(data['Age'], bins=[-np.inf, 18, 30, 45, np.inf], labels=['child', 'young', 'middle-aged', 'aged'])

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, random_state=7, stratify=data['AgeCat'])

import os

os.makedirs('data', exist_ok=True)

train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)


train_set, val_set = train_test_split(train, test_size=0.2, random_state=7, stratify=train['AgeCat'])
train_set.drop(columns=['AgeCat'], axis=1, inplace= True)
val_set.drop(columns=['AgeCat'], axis=1, inplace=True)

X_train = train_set.drop(columns=['Category'])
y_train = train_set['Category']
X_val = val_set.drop(columns=['Category'])
y_val = val_set['Category']

num_cols = X_train.select_dtypes(include='number').columns
cat_cols = X_train.select_dtypes(include='object').columns

from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])

X_val[num_cols] = num_imputer.transform(X_val[num_cols])
X_val[cat_cols] = cat_imputer.transform(X_val[cat_cols])

Q1 = data['Age'].quantile(0.25)
Q3 = data['Age'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['Age'] < lower_bound) | (data['Age'] > upper_bound)]
print(outliers)


from sklearn.preprocessing import StandardScaler, OrdinalEncoder

scaler = StandardScaler()
encoder = OrdinalEncoder()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])

X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_val[cat_cols] = encoder.transform(X_val[cat_cols])

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_val)

print(log_reg.score(X_val, y_val))
# Build a model using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(rf.score(X_val, y_val))

#Build a model using GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
print(gb.score(X_val, y_val))
#Based on the accuracy, the RandomForestClassifier model is selected as the final model
import joblib

os.makedirs('models', exist_ok=True)

joblib.dump(rf, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoder, 'models/encoder.pkl')
joblib.dump(num_imputer, 'models/num_imputer.pkl')
joblib.dump(cat_imputer, 'models/cat_imputer.pkl')