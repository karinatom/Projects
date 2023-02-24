import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Удаляем ненужные/лишние колонки
def filter_data(df):

    df = df.copy()
    columns_to_drop = [
       'id',
       'url',
       'region',
       'region_url',
       'price',
       'manufacturer',
       'image_url',
       'description',
       'posting_date',
       'lat',
       'long'
   ]

    return df.drop(columns_to_drop, axis=1)

# обработка выбросов
def year_outliers(df):

    df = df.copy()

    q25 = df.year.quantile(0.25)
    q75 = df.year.quantile(0.75)
    iqr = q75 - q25
    boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)

    df.loc[df['year'] < boundaries[0], 'year'] = round(boundaries[0])
    df.loc[df['year'] > boundaries[1], 'year'] = round(boundaries[1])

    return df
# создание новых фичей
def feature_engineering(df):

    df = df.copy()

    df.loc[:, 'short_model'] = df['model'].apply(lambda x: x.lower().split(' ')[0] if not pd.isna(x) else x) # Добавляем фичу "short_model" – это первое слово из колонки model
    df.loc[:, 'age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average')) # Добавляем фичу "age_category"

    return df


def main():

    print(f'Price category prediction pipeline')

    df = pd.read_csv('data/auto.csv')

    data_preparation = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('outliers', FunctionTransformer(year_outliers)),
        ('new_features', FunctionTransformer(feature_engineering))
    ]) # пайплайн подготовки и очистки данных

    df = data_preparation.transform(df)

    X = df.drop('price_category', axis=1) # переменные для предикторов и целевой переменной
    y = df['price_category']

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])  # пайплайн обработки числовых фичей

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])  # пайплайн обработки категориальных фичей

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ]) # объединим работу предыдущих пайплайнов

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    ) # список обучаемых моделей

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ]) # пайплайн, запускающий обучение моделей

        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy') # проверка моделей кросс-валидацией
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe
            # выбор лучшей модели

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {score.mean():.4f}')
    joblib.dump(best_pipe, 'price_category_pipe.pkl') # сохранение выбранной модели


if __name__ == '__main__':
    main()


