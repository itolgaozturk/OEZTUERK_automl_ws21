import os.path

import joblib
import optuna as optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from hpo_visualization import list_hypervolume_best_trials, plot_hyper_volume
from utils import scale_Xs, feature_filter, separate_X_y, data_loader
import matplotlib.pyplot as plt

def define_model(trial, X_train_filtered_scaled, y_train, X_val_filtered_scaled, y_val, seed):
    lr = trial.suggest_int("learning_rate", 1, 50)/100
    n_estimators = trial.suggest_int("n_estimators", 50, 150)
    subsample = trial.suggest_int("subsample", 5, 10)/10
    min_samples_split = trial.suggest_int("min_samples_split", 2, 300)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50)
    max_depth = trial.suggest_int("max_depth", 2, 50)
    min_impurity_decrease = trial.suggest_int("min_impurity_decrease", 1, 100)/100
    max_features = trial.suggest_int("max_features", 1, 100)/100
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 100, len(X_train_filtered_scaled))
    ccp_alpha = trial.suggest_int("ccp_alpha", 0, 100)/1000000

    # The HPs below are to finish the training earlier if there is no improvement,
    # so they are to decrease time, not to increase accuracy
    # So they can not make our objectives better
    # validation_fraction = trial.suggest_int("validation_fraction", 1, 10)/10
    # n_iter_no_change = trial.suggest_int("n_iter_no_change", 5, 100)
    # tol = trial.suggest_int("tol", 5, 100)

    clf = GradientBoostingClassifier(learning_rate=lr,
                                     n_estimators=n_estimators,
                                     subsample=subsample,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     max_depth=max_depth,
                                     min_impurity_decrease=min_impurity_decrease,
                                     max_features=max_features,
                                     max_leaf_nodes=max_leaf_nodes,
                                     ccp_alpha=ccp_alpha,
                                     random_state=seed).fit(X_train_filtered_scaled, y_train)

    mis_rate = 1 - clf.score(X_val_filtered_scaled, y_val)

    return clf, mis_rate

def objective(trial, X_train, y_train, X_val, y_val, seed):
    # filter the features
    X_train_filtered, X_val_filtered, num_features = feature_filter(True, X_train, y_train, X_val, trial)
    # scale the features
    X_train_filtered_scaled, X_val_filtered_scaled = scale_Xs(X_train_filtered, X_val_filtered)
    # apply Gradient Boosting with trial parameters
    model, mis_rate = define_model(trial, X_train_filtered_scaled, y_train, X_val_filtered_scaled, y_val, seed)
    relative_num_feat = num_features/len(X_train.columns)
    print("accuracy:", 1 - mis_rate)

    # multi-objective:
    # 1) relative num features
    # 2) misclassification rate
    return relative_num_feat, mis_rate

def automl(df, seed, time_budget, test_size, study_save_name=None):
    # create the validation set which we will tune the HPs on
    df_train, df_val = train_test_split(df, test_size=test_size, random_state=seed)
    X_train, y_train, X_val, y_val = separate_X_y(df_train, df_val)
    # assign the seed for reproducibility
    sampler = optuna.samplers.TPESampler(seed=seed)
    # both objectives shd be minimized
    study = optuna.create_study(sampler=sampler, directions=["minimize", "minimize"])
    # used lambda function to be able to provide multiple arguments to the objective function
    func = lambda trial: objective(trial, X_train, y_train, X_val, y_val, seed)
    study.optimize(func, timeout=time_budget, show_progress_bar=True)
    # you can save the study and load back again to avoid multiple runs
    if study_save_name != None:
        joblib.dump(study, os.path.join("results", study_save_name))
    # plot the trials and best trials
    plt.fig = optuna.visualization.matplotlib \
        .plot_pareto_front(study, include_dominated_trials=True,
                           target_names=["Relative Number of Features", "Misclassification Rate"])
    # get the best trial coordinates
    num_feature_trials, mis_class_trials = list_hypervolume_best_trials(study)
    # plot hypervolume using best_trial coordinates
    plot_hyper_volume(x=num_feature_trials, y=mis_class_trials)
    return study

if __name__ == '__main__':
    seed = 27
    filename = "madelon.arff"
    time_budget = 30
    test_size = 0.3
    study_save_name = None
    df = data_loader("madelon.arff", seed)
    study = automl(df, seed, time_budget, test_size, study_save_name)
    plt.show()