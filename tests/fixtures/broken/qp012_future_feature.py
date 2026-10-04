from sklearn.linear_model import LinearRegression


def research(df):
    df["fwd_return"] = df["close"].pct_change().shift(-5)
    features = df[["ret_1", "fwd_return"]].dropna()
    model = LinearRegression()
    model.fit(features, df.loc[features.index, "label"])
    return model
