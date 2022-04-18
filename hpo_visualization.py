import joblib
import optuna as optuna
import numpy as np
import matplotlib.pyplot as plt

# credits to https://stackoverflow.com/questions/42692921/how-to-create-hypervolume-and-surface-attainment-plots-for-2-objectives-using
def plot_hyper_volume(x, y):
    # convert them to np array
    x = np.array(x)
    y = np.array(y)

    # Zip x and y into a numpy ndarray
    # sort them wrt x (left-to-right)
    coordinates = np.array(sorted(zip(x, y)))

    # create empty pareto set
    pareto_set = np.full(coordinates.shape, np.inf)

    # add pareto-front points to pareto-set
    i = 0
    for point in coordinates:
        if i == 0:
            pareto_set[i] = point
            i += 1
        # compare y coordinate whether its the most bottom point till now
        elif point[1] < pareto_set[:, 1].min():
            pareto_set[i] = point
            i += 1

    # Get rid of unused spaces
    pareto_set = pareto_set[:i + 1, :]

    # create reference point
    reference_point = np.array([[pareto_set[:i, 0].max(), pareto_set[:i, 1].max()]])

    # Add the reference point to the pareto set
    pareto_set = np.concatenate((pareto_set, reference_point))

    # These points will define the path to be plotted and filled
    x_path_of_points = []
    y_path_of_points = []

    for index, point in enumerate(pareto_set):
        if index < i - 1:
            # horizontal line
            plt.plot([point[0], pareto_set[index + 1][0]], [point[1], point[1]],
                     c='#ff0000', marker = 'o',markersize=4, markevery=[0])
            # vertical line
            plt.plot([pareto_set[index + 1][0], pareto_set[index + 1][0]], [point[1], pareto_set[index + 1][1]],
                     c='#ff0000')
            if index != 0:
                x_path_of_points += [point[0], point[0], pareto_set[index + 1][0]]
                y_path_of_points += [pareto_set[index - 1][1], point[1], point[1]]

    # Link 1 to Reference Point from leftest best-trial (most-upper line)
    plt.plot([pareto_set[0][0], reference_point[0][0]], [pareto_set[0][1], reference_point[0][1]], c='#4270b6',
             marker = 'o',markersize=8, markevery=[1])
    # Link 2 to Reference Point from rightest best-trial (most-right line)
    plt.plot([pareto_set[-2][0], reference_point[0][0]], [pareto_set[-2][1], reference_point[0][1]], c='#4270b6')

    # Highlight the Last Point
    plt.plot(pareto_set[-2][0], pareto_set[-2][1], 'o', c='#ff0000', markersize=4)

    # Fill the area between the Pareto set and Ref y
    if len(x_path_of_points)>0 and len(y_path_of_points)>0:
        plt.fill_betweenx(y_path_of_points, x_path_of_points, max(x_path_of_points) * np.ones(len(x_path_of_points)),
                          color='#dfeaff', alpha=1)

    plt.tight_layout()

def list_hypervolume_best_trials(study):
    # create a list which includes best_trial points' coordinates
    num_feature_trials=[]
    mis_class_trials=[]
    study_best_trials = study.best_trials

    for trial_no in range(len(study_best_trials)):
        trial_values = study_best_trials[trial_no].values
        num_feature_trials.append(trial_values[0])
        mis_class_trials.append(trial_values[1])

    return num_feature_trials, mis_class_trials

if __name__ == '__main__':
    # to try visualizations
    #study1 = joblib.load('results/madelon_720_cv_1.pkl')
    study2 = joblib.load('results/madelon_720_cv_2.pkl')

    #fig = optuna.visualization.plot_param_importances(study1, target=lambda t: t.values[1], target_name="mis_rate")
    fig2 = optuna.visualization.plot_param_importances(study2, target=lambda t: t.values[1], target_name="Misclassification Rate")

    #fig = optuna.visualization.plot_slice(study1, target=lambda t: t.values[1])

    #fig = optuna.visualization.plot_contour(study1, target=lambda t: t.values[1])

    fig2.show()

    #plt.fig = optuna.visualization.matplotlib.plot_pareto_front(study, include_dominated_trials=True,
     #                                                           target_names=["num_features", "mis_rate"])
    #num_feature_trials, mis_class_trials = list_hypervolume_best_trials(study)
    #plot_hyper_volume(x=num_feature_trials, y=mis_class_trials)
    #plt.show()