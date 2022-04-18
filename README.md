# automl_exam
Optimizing Gradient Boosting Hyperparameters using Bayesian Optimisation

I also added untuned Gradient Boosting model to compare with tuned version

The only parameters you need to give:

Parameter | Description
--- | ---
seed | for reproducibility (I used 27) 
filename | "madeline.arff" or "madelon.arff"
time_budget | HPO runtime (per CV fold), please give an appropriate number to obtain at least 30 iterations for madelon, 45 iterations for madeline
val_size_hpo | size of the validation set over size of the train+validation set 
study_save_name | if you want to save the study for future use provide a name, otherwise give None

## My Experiment Results:

### On Madelon

time_budget is used as 30 seconds and val_size_hpo is used as 0.3

Model | Misclassification Rate
--- | ---
Random Forest | 0.280
LR+SVM+NB | 0.417
(extra) GB without AutoML | 0.247
AutoML | 0.169

### On Madeline

time_budget is used as 300 seconds/fold (around 250 trials/fold) and val_size_hpo is used as 0.3

Model | Misclassification Rate
--- | ---
Random Forest | 0.227
LR+SVM+NB | 0.408
(extra) GB without AutoML | 0.224
AutoML | 0.138

I provided the datasets in case there would be an update in the online version.
I also provided my resulting files in case you don't want to rerun

### An example resulting graph
Plot of an pareto front, also including the hypervolume plot

![alt text](example.png)