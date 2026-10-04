from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

PARAMS = {
    "n_estimators": [50, 100, 200, 400, 800],
    "max_depth": [2, 3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.1, 0.3],
    "subsample": [0.5, 0.8, 1.0],
}


def research(X, y):
    search = GridSearchCV(GradientBoostingClassifier(), PARAMS, cv=TimeSeriesSplit(5))
    return search.fit(X, y)
