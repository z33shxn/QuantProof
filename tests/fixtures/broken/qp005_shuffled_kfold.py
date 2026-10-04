from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score


def research(X, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    return cross_val_score(Ridge(), X, y, cv=cv).mean()
