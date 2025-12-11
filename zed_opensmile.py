# %%
# import the usual libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report

from ztime import helper

# %%
#ignore warnings
import warnings
warnings.simplefilter(action='ignore')

# %%
# import the .csv files in merged_openface_out as dataframes in a dictionary
import os
import glob

# get the current directory
path = os.getcwd()

print(path)
# get the path to the directory with the csv files
path = path + '/merged_opensmile_out'
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

path = path + '/opensmile_out_A'
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
# read the full_dataset.csv file into a dataframe. Keep only the 'Dyad Number' and 'Truth/Lie' columns
full_dataset = pd.read_csv('full_dataset.csv', usecols=['Dyad Number', 'Truth/Lie'])
# delete the duplicates in the full_dataset dataframe based on the 'Dyad Number' column
full_dataset = full_dataset.drop_duplicates(subset='Dyad Number')

# %%
# Create a function to transform the dataframes in a dictionary into a single 3d numpy array, structured as (n_samples, n_features, n_timepoints).
# Use the keys of the dictionary, as integers, from the smallest to the largest, as the first dimension of the numpy array.
#Use the columns of the dataframes as the second dimension of the numpy array.
# Use the rows of the dataframes as the third dimension of the numpy array.

def dict_to_array(data):
    # get the keys of the dictionary
    keys = list(data.keys())
    # transform the keys into integers
    keys = [int(key) for key in keys]
    # sort the keys
    keys.sort()
    # transform the keys back into strings
    keys = [str(key) for key in keys]
    # print the keys
    print(keys)
    # get the number of keys
    n_keys = len(keys)
    # get the number of columns
    n_columns = data[keys[0]].shape[1]
    # get the number of rows
    n_rows = data[keys[0]].shape[0]
    # create an empty numpy array
    array = np.zeros((n_keys, n_columns, n_rows))
    # loop through the keys
    for i in range(n_keys):
        # get the key
        key = keys[i]
        # get the dataframe
        df = data[key]
        # get the values of the dataframe
        values = df.values
        # store the values in the numpy array
        array[i, :, :] = values.T
    return array

# %%
# transform the dataframes in the dictionary into a single 3d numpy array
X = dict_to_array(data)

X_A = dict_to_array(data_A)

# create a label array, there 'Lie' is 0 and 'Truth' is 1
y = full_dataset['Truth/Lie'].values
y = np.where(y == 'Lie', 0, 1)

# %%
# create a function to perform the training using leave one out cross validation and to create the confusion matrix and the classification report
def train_one_out(X, y):
    # create a leave one out cross validation object
    loo = LeaveOneOut()
    # create an empty list to store the predictions
    predictions = []
    # loop through the training and test sets
    for i, (train_index, test_index) in enumerate(loo.split(X)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        
        pred, time = helper.simpleTrial(X_train=X[train_index], y_train=y[train_index], X_test=X[test_index], y_test=y[test_index])

        # store the prediction
        predictions.append(pred)
    # create the confusion matrix
    cm = confusion_matrix(y, predictions)
    # create the classification report
    cr = classification_report(y, predictions)
    return cm, cr, predictions

# %%
cm_Z, cr_Z, predictions_Z = train_one_out(X, y)

# %%
cm_Z_A, cr_Z_A, predictions_Z_A = train_one_out(X_A, y)

# %%
# create a file to store the results

with open('results_Z.txt', 'a') as f:
    f.write('Z time on merged_opensmile_out\n')
    f.write('Confusion Matrix:\n')
    f.write(str(cm_Z))
    f.write('\n')
    f.write('Classification Report:\n')
    f.write(cr_Z)
    f.write('\n')
    f.write('Z time on opensmile_out_A\n')
    f.write('Confusion Matrix:\n')
    f.write(str(cm_Z_A))
    f.write('\n')
    f.write('Classification Report:\n')
    f.write(cr_Z_A)
    f.write('\n')


