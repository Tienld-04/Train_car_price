from datetime import datetime
import numpy as np
import pandas as pd
# from ydata_profiling import ProfileReport
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,  OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lazypredict.Supervised import LazyRegressor
import joblib
from Model_Linear import MyLinearRegression
from Model_Decision_tree import MyDecisionTreeRegressor



data_path = "D:\\Python\\API_model\\ML3\\car_price\\car_resale_prices.csv"

data = pd.read_csv(data_path)
# profile = ProfileReport(data, title= "Car price", explorative=True)
# profile.to_file('car_price.html')
data = data.drop('Unnamed: 0', axis=1)


def covert_columns(data):
    current_year = datetime.now().year
    def process_name(value):
        if not isinstance(value, str):
            return pd.Series([np.nan, np.nan, np.nan])
        parts = value.split()
        if len(parts) < 2:
            return pd.Series([parts[0], np.nan, np.nan])
        model = parts[1]
        variant = ' '.join(parts[2:]) if len(parts) > 2 else np.nan
        return pd.Series([model, variant])

    def clean_resale_price(value):
        try :
            if isinstance(value, str):
                value = value.replace('₹', '').replace('Lakh', '').strip()
                return float(value) * 100000
        except :
            return np.nan
        return np.nan

    def clean_engine_capacity(value) :
        try :
            if isinstance(value, str) :
                value = value.replace('cc', '').strip()
                return int(value)
        except :
            return np.nan
        return np.nan

    def clean_kms_driven(value):
        try :
            if isinstance(value, str) :
                value = value.replace('Kms', "").replace(',', '').strip()
                return int(value)
        except :
            return np.nan
        return np.nan

    def clean_max_power(value):
        try :
            if isinstance(value, str) :
                value = value.replace('bhp', '').strip()
                return float(value)
        except :
            return np.nan
        return np.nan
    def clean_mileage(value):
        try :
            if isinstance(value, str):
                value = value.replace('kmpl', '').strip()
                return float(value)
        except :
            return np.nan
        return np.nan
    data[['model','variant' ]] = data['full_name'].apply(process_name)
    data['registered_year'] = pd.to_numeric(data['registered_year'], errors='coerce')
    data['age_car'] = data['registered_year'].apply(
        lambda x : current_year - x if pd.notnull(x) else np.nan
    )
    data_ = data.drop('full_name', axis=1)
    data_ = data_.drop('registered_year', axis=1)
    cols = ['model', 'variant', 'age_car'] + [col for col in data_.columns if col not in ['model', 'variant', 'age_car']]
    data_['resale_price'] = data_['resale_price'].apply(clean_resale_price)
    data_['engine_capacity'] = data_['engine_capacity'].apply(clean_engine_capacity)
    data_['kms_driven'] = data_['kms_driven'].apply(clean_kms_driven)
    data_['max_power'] = data_['max_power'].apply(clean_max_power)
    data_['mileage'] = data_['mileage'].apply(clean_mileage)
    return data_[cols]
data = covert_columns(data)
data = data.dropna(subset=['resale_price'])

target = 'resale_price'
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1111)

#transform

list_col_num = ['age_car',  'engine_capacity', 'kms_driven','max_power', 'seats', 'mileage']
list_col_nom = ['model','variant', 'insurance','transmission_type', 'owner_type','fuel_type', 'body_type', 'city']
num_transformer = Pipeline(steps=[
   ( 'imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])

nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False,handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num_feature', num_transformer, list_col_num),
    ('nom_feature', nom_transformer, list_col_nom)
])
#
# x_train = preprocessor.fit_transform(x_train)
# x_test = preprocessor.transform(x_test)


# reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
# models, predictions = reg.fit(x_train, x_test, y_train, y_test)


reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", MyDecisionTreeRegressor(max_depth=10))
    #("regressor", MyLinearRegression())
])
# params = {
#     "regressor__kernel": ['rbf', 'linear', 'poly', 'sigmoid'],
#     "regressor__C": [0.1, 1, 10, 100],
#     "regressor__epsilon": [0.01, 0.1, 0.2, 0.5],
#     "regressor__gamma": ['scale', 'auto', 0.01, 0.1, 1],
#     "regressor__degree": [2, 3, 4],  # chỉ áp dụng cho kernel='poly'
#     "preprocessor__num_feature__imputer__strategy": ['mean', 'median']
# }
# model = RandomizedSearchCV(
# estimator=reg,
#     param_distributions=params,
#     n_iter=30,
#     scoring="r2",
#     cv=6,
#     verbose=2,
#     n_jobs=6
# )
# model.fit(x_train, y_train)
# print(model.best_score_)
# print(model.best_params_)

reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)


print("R2 score:",r2_score(y_test, y_predict))
print("MAE {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE {}".format(mean_squared_error(y_test, y_predict)))
print("RMSE {}".format(root_mean_squared_error(y_test, y_predict)))

model_path = "D:\\Python\\API_model\\ML3\\car_price\\trained_car_price_model_Linear.pkl"
joblib.dump(reg, model_path)
print("Model saved to {}".format(model_path))
