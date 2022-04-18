from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from automl import automl
from utils import scale_Xs, feature_filter, data_loader, separate_X_y
import matplotlib.pyplot as plt

def find_best_trial(study_trials):
    best_trial = None
    min_misclass = 1
    for trial_no in range(len(study_trials)):
        trial_values = study_trials[trial_no].values
        if trial_values[1] < min_misclass:
            min_misclass = trial_values[1]
            best_trial = study_trials[trial_no]
    return best_trial

def comparison_models(X_train_scaled, y_train, X_test_scaled, y_test, seed):
    acc_rf = RandomForestClassifier(min_samples_split=0.01, random_state=seed) \
        .fit(X_train_scaled, y_train).score(X_test_scaled, y_test)

    clf_lr = LogisticRegression(random_state=seed)
    clf_svm = SVC(random_state=seed)
    clf_nb = GaussianNB()
    acc_ensemble = VotingClassifier(estimators=[('clf_lr', clf_lr), ('clf_svm', clf_svm), ('clf_nb', clf_nb)]) \
        .fit(X_train_scaled, y_train).score(X_test_scaled, y_test)

    acc_GB_without_AutoML = GradientBoostingClassifier(min_samples_split=0.01, random_state=seed) \
        .fit(X_train_scaled, y_train).score(X_test_scaled, y_test)
    return acc_rf, acc_ensemble, acc_GB_without_AutoML

def optimized_HP_refit(best_trial, X_train_filtered_scaled, y_train, X_test_filtered_scaled, y_test, seed):
    best_trial_GB_params = best_trial.params.copy()
    del best_trial_GB_params["feat_threshold"]
    best_trial_GB_params["learning_rate"] = best_trial_GB_params["learning_rate"] / 100
    best_trial_GB_params["subsample"] = best_trial_GB_params["subsample"] / 10
    best_trial_GB_params["min_impurity_decrease"] = best_trial_GB_params["min_impurity_decrease"] / 100
    best_trial_GB_params["max_features"] = best_trial_GB_params["max_features"] / 100
    best_trial_GB_params["ccp_alpha"] = best_trial_GB_params["ccp_alpha"] / 1000000

    acc_AutoML = GradientBoostingClassifier(random_state=seed).set_params(**best_trial_GB_params) \
        .fit(X_train_filtered_scaled, y_train).score(X_test_filtered_scaled, y_test)

    return acc_AutoML

def main(seed, filename, time_budget, val_size_hpo, study_save_name=None):
    df = data_loader(filename, seed)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    splits = kfold.split(df, df['class'])

    acc_rf_sum = 0
    acc_ensemble_sum = 0
    acc_GB_without_AutoML_sum = 0
    acc_AutoML_sum = 0
    num_features_optimized_sum = 0
    for n, (train_index, test_index) in enumerate(splits):
        print("\n---------------------------------------------------------------------CV FOLD " + str(n+1) + "----------------------------------------------------------------------------------------------")
        df_train = df.iloc[train_index, :]
        df_test = df.iloc[test_index, :]

        X_train, y_train, X_test, y_test = separate_X_y(df_train, df_test)

        X_train_scaled, X_test_scaled = scale_Xs(X_train, X_test)

        acc_rf, acc_ensemble, acc_GB_without_AutoML = comparison_models(X_train_scaled, y_train,
                                                                        X_test_scaled, y_test, seed)
        acc_rf_sum += acc_rf
        acc_ensemble_sum += acc_ensemble
        acc_GB_without_AutoML_sum += acc_GB_without_AutoML

        if study_save_name == "auto":
            study_save_name = filename[:-5] + "_" + str(time_budget) + "_cv_" + str(n + 1) + ".pkl"
        study = automl(df_train, seed, time_budget, val_size_hpo, study_save_name)
        study_save_name = "auto"
        plt.title("Pareto Front: CV Fold " + str(n+1))

        # Part to use our study findings on testing data
        best_trial = find_best_trial(study.best_trials)

        num_features_optimized = best_trial.params["feat_threshold"]
        num_features_optimized_sum += num_features_optimized

        X_train_filtered, X_test_filtered, _ = feature_filter(False, X_train, y_train,
                                                              X_test, feat_threshold=num_features_optimized)
        X_train_filtered_scaled, X_test_filtered_scaled = scale_Xs(X_train_filtered, X_test_filtered)

        acc_AutoML_sum += optimized_HP_refit(best_trial, X_train_filtered_scaled, y_train,
                                        X_test_filtered_scaled, y_test, seed)

    plt.show()
    print("\n\t\tMisclassification Rates")
    print("Random Forest:", 1-acc_rf_sum/10)
    print("LR+SVM+NB:", 1-acc_ensemble_sum/10)
    print("GB without AutoML:", 1-acc_GB_without_AutoML_sum/10)
    print("AutoML:", 1-acc_AutoML_sum/10)
    print("Relative number of features (Optimized):", num_features_optimized_sum/len(df.columns)/10)

if __name__ == "__main__":
    # ps run for at least 30 secs for each CV fold
    seed = 27
    filename = "madeline.arff"
    time_budget = 720
    val_size_hpo = 0.3
    # use "auto" for study_save_name if needed
    study_save_name = "auto"
    main(seed, filename, time_budget, val_size_hpo, study_save_name)