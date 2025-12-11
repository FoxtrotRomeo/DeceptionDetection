# %%
# import the usual libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, classification_report
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.classification.kernel_based import RocketClassifier

# %%
#ignore warnings
import warnings
warnings.simplefilter(action='ignore')

# %%
# import the .csv files in merged_opensmile_out as dataframes in a dictionary
import os
import glob

# get the current directory
path = os.getcwd()

print(path)
# get the path to the directory with the csv files
path = path + '/merged_opensmile_out_cut'
# get the list of files in the directory
all_files = glob.glob(path + "/*.csv")

# create an empty dictionary to store the dataframes
data = {}
# loop through the list of files
for filename in all_files:
    # get the name of the file
    name = os.path.basename(filename)
    # delete the .csv extension
    name = name[:-4]
    # read the file into a dataframe
    df = pd.read_csv(filename, index_col='Unnamed: 0', header=0)
    # drop the columns starting with timestamp
    df = df.drop(df.filter(regex='timestamp').columns, axis=1)
    # store the dataframe in the dictionary
    data[name] = df

# get the path to the directory with the csv files
path = os.getcwd()

path = path + '/opensmile_out_A_cut'
# get the list of files in the directory
all_files = glob.glob(path + "/*.csv")
print(all_files)
# create an empty dictionary to store the dataframes
data_A = {}
# loop through the list of files
for filename in all_files:
    # get the name of the file
    name = os.path.basename(filename)
    # delete the .csv extension
    name = name[:-4]
    # read the file into a dataframe
    df = pd.read_csv(filename, index_col='Unnamed: 0', header=0)
    # drop the columns starting with timestamp
    df = df.drop(df.filter(regex='timestamp').columns, axis=1)
    # store the dataframe in the dictionary
    data_A[name] = df

# %%
# check the number of missing values in data and in data_A
missing_values = {}
for key in data.keys():
    missing_values[key] = data[key].isnull().sum().sum()
missing_values_A = {}
for key in data_A.keys():
    missing_values_A[key] = data_A[key].isnull().sum().sum()

print(missing_values)
print(missing_values_A)

# %%
sorted_keys = sorted(list(data.keys()))
print(sorted_keys)
# create a list of groups, where each group is given by the elements of sorted_keys, except the last two characters
groups = list(set([key[:-2] for key in sorted_keys]))
print(sorted(groups))

group_dict = {}
for group in groups:
    # create a list of keys for the current group
    keys = [key for key in sorted_keys if key[:-2] == group]
    # create a list of dataframes for the current group
    dfs = [data[key] for key in keys]

    # append the list of dataframes to the dictionary
    group_dict[group] = dfs

# %%
print(len(group_dict['7']))

# %%
# read the full_dataset.csv file into a dataframe. Keep only the 'Dyad Number' and 'Truth/Lie' columns
full_dataset = pd.read_csv('full_dataset.csv', usecols=['Dyad Number', 'Truth/Lie'])
# delete the duplicates in the full_dataset dataframe based on the 'Dyad Number' column
full_dataset = full_dataset.drop_duplicates(subset='Dyad Number')

# %%
print(type(data['1_1']))

# %%
# Create a function to transform the dataframes in a dictionary into a single 3d numpy array, structured as (n_samples, n_features, n_timepoints).
# Use the keys of the dictionary, as integers, from the smallest to the largest, as the first dimension of the numpy array.
#Use the columns of the dataframes as the second dimension of the numpy array.
# Use the rows of the dataframes as the third dimension of the numpy array.

def dict_to_array(data):
    # get the number of keys
    n_keys = len(list(data.keys()))
    # get the number of columns
    n_columns = data[list(data.keys())[0]].shape[1]
    # get the number of rows
    n_rows = data[list(data.keys())[0]].shape[0]
    # create an empty numpy array
    array = np.zeros((n_keys, n_columns, n_rows))
    # create an empty list to store the groups from the keys
    groups = np.array([])
    # loop through the keys
    for i in range(n_keys):
        # get the key
        key = list(data.keys())[i]
        # get the group: the key except the last two characters
        group = key[:-2]
        # append the group to the list
        groups = np.append(groups, int(group))
        df = data[key]
        # get the values of the dataframe
        values = df.values
        # store the values in the numpy array
        array[i, :, :] = values.T
    for element in groups:
        element = int(element)
    return array, groups

# %%
# transform the dataframes in the dictionary into a single 3d numpy array
print('working on X')
X, groups_X = dict_to_array(data)
print('working on X_A')
X_A, groups_XA = dict_to_array(data_A)

# create a label array, there 'Lie' is 0 and 'Truth' is 1
y = full_dataset['Truth/Lie'].values
y = np.where(y == 'Lie', 0, 1)

# %%
for element in groups_X:
    element = int(element)

# %%
# create a dictionary using the "Dyad Number" column of the full_dataset dataframe as keys and the "Truth/Lie" column as values, where 'Lie' is 0 and 'Truth' is 1
map = full_dataset.set_index('Dyad Number').to_dict()['Truth/Lie']
# change each truth value in map to 1 and each lie value to 0
for key in map.keys():
    map[key] = 1 if map[key] == 'Truth' else 0

# create a numpy array mapping each value in groups to the corresponding value in map
y = np.array([map[group] for group in groups_X])
y_a = np.array([map[group] for group in groups_XA])

# %%
# create a canonical interval forest model
cif = CanonicalIntervalForest(n_estimators=100, random_state=47, n_jobs=-1)

# create a rocket model
rocket = RocketClassifier(num_kernels=1000, random_state=47, n_jobs=-1)

# %%
# create a function to perform the training using leave one out cross validation and to create the confusion matrix and the classification report
def train_one_out(X, y, model):
    # create a leave one out cross validation object
    logo = LeaveOneGroupOut()
    # create an empty list to store the predictions
    predictions = []
    # loop through the training and test sets
    for i, (train_index, test_index) in enumerate(logo.split(X, y, groups=groups_X)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        # train the cif model
        model.fit(X[train_index], y[train_index])
        # the test set contains multiple samples. Produce a prediction for each sample
        for j in range(len(test_index)):
            # get the test sample
            X_test = X[test_index][j]
            # reshape X_test to 1, dimnsion 0, dimension 1
            X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
            y_pred = model.predict(X_test)
            # store the prediction
            predictions.append(y_pred)
        print(len(predictions))
    print('for loop done')
    # create the confusion matrix
    cm = confusion_matrix(y, predictions)
    # create the classification report
    cr = classification_report(y, predictions)
    return cm, cr, predictions

# %%
cm_rocket, cr_rocket, predictions_rocket = train_one_out(X, y, rocket)

# %%
cm_cif, cr_cif, predictions_cif = train_one_out(X, y, cif)

# %%
cm_rocket_A, cr_rocket_A, predictions_rocket_A = train_one_out(X_A, y, rocket)

# %%
cm_cif_A, cr_cif_A, predictions_cif_A = train_one_out(X_A, y, cif)

# %%
# create a file to store the results

with open('results.txt', 'a') as f:
    f.write('Rocket on merged_opensmile_out_cut\n')
    f.write('Confusion Matrix:\n')
    f.write(str(cm_rocket))
    f.write('\n')
    f.write('Classification Report:\n')
    f.write(cr_rocket)
    f.write('\n')
    f.write('Rocket on opensmile_out_A_cut\n')
    f.write('Confusion Matrix:\n')
    f.write(str(cm_rocket_A))
    f.write('\n')
    f.write('Classification Report:\n')
    f.write(cr_rocket_A)
    f.write('\n')
    f.write('CIF on merged_opensmile_out_cut\n')
    f.write('Confusion Matrix:\n')
    f.write(str(cm_cif))
    f.write('\n')
    f.write('Classification Report:\n')
    f.write(cr_cif)
    f.write('\n')
    f.write('CIF on opensmile_out_A_cut\n')
    f.write('Confusion Matrix:\n')
    f.write(str(cm_cif_A))
    f.write('\n')
    f.write('Classification Report:\n')
    f.write(cr_cif_A)
    f.write('\n')


