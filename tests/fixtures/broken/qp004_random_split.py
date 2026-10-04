from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def research(df):
    X = df[["ret_1", "ret_5"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression().fit(X_train, y_train)
    return model.score(X_test, y_test)
