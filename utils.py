from scipy.io import arff
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_Xs(X_train, X_test):
    # standardize the features (train and test are done independently)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    return X_train_scaled, X_test_scaled

def feature_filter(hpo, X_train, y_train, X_test, trial=None, feat_threshold=None):
    if hpo == True:
        feat_threshold = trial.suggest_int("feat_threshold", 1, len(X_train.columns))

    # Select K-best features where K shd be optimized
    feat_selector = SelectKBest(k=feat_threshold).fit(X_train, y_train)
    feat_filter = feat_selector.get_support(indices=True)

    # use the same features on test set
    X_train_filtered = X_train.iloc[:, feat_filter]
    X_test_filtered = X_test.iloc[:, feat_filter]

    return X_train_filtered, X_test_filtered, feat_threshold

def data_loader(filename, seed, frac=1):
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    if filename == "madelon.arff":
        df['class'] = df['Class'].str.decode("utf-8")
        df = df.drop("Class", axis=1)
    elif filename == "madeline.arff":
        df['Class'] = df['class'].str.decode("utf-8")
    else:
        print("Please give either madeline or madelon datasets")
    if frac < 1:
        # to use a sample of the data (e.g. multi-fidelity)
        df = df.sample(frac=frac, random_state=seed)
    return df

def separate_X_y(df_train, df_test):
    # separate features and the target
    y_train = df_train['class'].copy()
    X_train = df_train.drop("class", axis=1)
    y_test = df_test['class'].copy()
    X_test = df_test.drop("class", axis=1)
    return X_train, y_train, X_test, y_test