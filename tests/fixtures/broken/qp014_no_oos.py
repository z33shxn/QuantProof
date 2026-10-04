from sklearn.ensemble import RandomForestRegressor


def research(X, y):
    model = RandomForestRegressor(n_estimators=200)
    model.fit(X, y)
    return model.score(X, y)
