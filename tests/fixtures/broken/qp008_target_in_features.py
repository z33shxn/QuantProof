from sklearn.ensemble import RandomForestClassifier


def research(df, X_test):
    y = df["target"]
    X = df[["ret_1", "vol_20", "target"]]
    model = RandomForestClassifier().fit(X, y)
    return model
