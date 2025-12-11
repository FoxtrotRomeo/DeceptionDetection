# %%
# import the usual libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.classification.kernel_based import RocketClassifier

# %%
#ignore warnings
import warnings
warnings.simplefilter(action='ignore')

# %%
# import the .csv files in merged_openface_out as dataframes in a dictionary
import os
import glob
print('Early Fusion')
# get the current directory
path = os.getcwd()

print(path)
# get the path to the directory with the csv files
path = path + '/merged_openface_out'
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

path = path + '/openface_out_A'
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

# get the current directory
path = os.getcwd()

print(path)
# get the path to the directory with the csv files
path = path + '/merged_opensmile_out'
# get the list of files in the directory
all_files = glob.glob(path + "/*.csv")

# create an empty dictionary to store the dataframes
data_voice = {}
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
    data_voice[name] = df

# get the current directory
path = os.getcwd()

print(path)
# get the path to the directory with the csv files
path = path + '/opensmile_out_A'
# get the list of files in the directory
all_files = glob.glob(path + "/*.csv")

# create an empty dictionary to store the dataframes
data_voice_A = {}
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
    data_voice_A[name] = df

# %%
# create a function that takes a dictionary of dataframes and returns a dictionary of dataframes in which each dataframe has been shortened by averaging the values in each column over a window of size window_size
def average_over_window(data, window_size):
    # create an empty dictionary to store the shortened dataframes
    data_short = {}
    # loop through the dataframes in the input dictionary
    for key in data:
        # get the dataframe
        df = data[key]
        # create an empty dataframe to store the shortened version
        df_short = pd.DataFrame()
        # loop through the columns in the dataframe
        for col in df.columns:
            # create an empty list to store the averaged values
            avg = []
            # loop through the values in the column
            for i in range(0, len(df[col]), window_size):
                # get the average of the values in the window
                avg.append(np.mean(df[col][i:i+window_size]))
            # add the averaged values to the shortened dataframe
            df_short[col] = avg
        # store the shortened dataframe in the output dictionary
        data_short[key] = df_short
    return data_short

# %%
from sktime.transformations.series.paa import PAA

paa = PAA(frames=10000)

data_short = {}
data_short_A = {}
data_short_voice = {}
data_short_voice_A = {}

for key in data.keys():
    data_short[key] = paa.fit_transform(data[key])
    data_short_A[key] = paa.fit_transform(data_A[key])
    data_short_voice[key] = paa.fit_transform(data_voice[key])
    data_short_voice_A[key] = paa.fit_transform(data_voice_A[key])

# %%
# save the dataframes in the three dictionaries as .csv files
save_path = 'data_short/'
save_path_A = 'data_short_A/'
save_path_voice = 'data_short_voice/'
save_path_voice_A = 'data_short_voice_A/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_path_A):
    os.makedirs(save_path_A)
if not os.path.exists(save_path_voice):
    os.makedirs(save_path_voice)
if not os.path.exists(save_path_voice_A):
    os.makedirs(save_path_voice_A)

for key in data_short.keys():
    data_short[key].to_csv(save_path + key + '.csv')
    data_short_A[key].to_csv(save_path_A + key + '.csv')
    data_short_voice[key].to_csv(save_path_voice + key + '.csv')
    data_short_voice_A[key].to_csv(save_path_voice_A + key + '.csv')

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
X = dict_to_array(data_short)

X_A = dict_to_array(data_short_A)

X_Voice = dict_to_array(data_short_voice)

X_Voice_A = dict_to_array(data_short_voice_A)

# create a label array, there 'Lie' is 0 and 'Truth' is 1
y = full_dataset['Truth/Lie'].values
y = np.where(y == 'Lie', 0, 1)

# %%
# create a function to perform the training using leave one out cross validation and to create the confusion matrix and the classification report
def train_one_out(X, y, model):
    # create a leave one out cross validation object
    loo = LeaveOneOut()
    # create an empty list to store the predictions
    predictions = []
    # loop through the training and test sets
    for i, (train_index, test_index) in enumerate(loo.split(X)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        # train the cif model
        model.fit(X[train_index], y[train_index])
        # produce a probability prediction
        y_pred_test = model.predict(X[test_index])
        # store the prediction
        predictions.append(y_pred_test)
    print(predictions)
    # create the confusion matrix
    cm = confusion_matrix(y, predictions)
    # create the classification report
    cr = classification_report(y, predictions)
    return cm, cr, predictions

# %%

# create a rocket model
rocket = RocketClassifier(num_kernels=1000, random_state=47, n_jobs=-1)

# %%
X_merged = np.concatenate((X, X_Voice), axis=1)
X_merged_a = np.concatenate((X_A, X_Voice), axis=1)
X_merged_v = np.concatenate((X, X_Voice_A), axis=1)
X_merged_va = np.concatenate((X_A, X_Voice_A), axis=1)

cm_rocket, cr_rocket, predictions_rocket = train_one_out(X_merged, y, rocket)
cm_rocket_a, cr_rocket_a, predictions_rocket_a = train_one_out(X_merged_a, y, rocket)
cm_rocket_v, cr_rocket_v, predictions_rocket_v = train_one_out(X_merged_v, y, rocket)
cm_rocket_va, cr_rocket_va, predictions_rocket_va = train_one_out(X_merged_va, y, rocket)

# %%
from sktime.transformations.series.sax import SAX
# apply the SAX transformation to the data, only to the last dimension of the numpy array
sax = SAX(word_size=10000, alphabet_size=6)
X = sax.fit_transform(X)
X_A = sax.fit_transform(X_A)
X_Voice = sax.fit_transform(X_Voice)
X_Voice_A = sax.fit_transform(X_Voice_A)

# %%
X_merged_sax = np.concatenate((X, X_Voice), axis=1)
X_merged_a_sax = np.concatenate((X_A, X_Voice), axis=1)
X_merged_v_sax = np.concatenate((X, X_Voice_A), axis=1)
X_merged_va_sax = np.concatenate((X_A, X_Voice_A), axis=1)

cm_rocket_sax, cr_rocket_sax, predictions_rocket_sax = train_one_out(X_merged_sax, y, rocket)
cm_rocket_a_sax, cr_rocket_a_sax, predictions_rocket_a_sax = train_one_out(X_merged_a_sax, y, rocket)
cm_rocket_v_sax, cr_rocket_v_sax, predictions_rocket_v_sax = train_one_out(X_merged_v_sax, y, rocket)
cm_rocket_va_sax, cr_rocket_va_sax, predictions_rocket_va_sax = train_one_out(X_merged_va_sax, y, rocket)

# %%
# create a file to store the results

with open('results_ef2.txt', 'a') as f:
    f.write('Rocket on merged_opensmile_out and merged_openface_out\n')
    f.write('Confusion Matrix:\n')
    f.write(str(cm_rocket))
    f.write('\n')
    f.write('Classification Report:\n')
    f.write(cr_rocket)
    f.write('\n')
    f.write('Rocket on merged_opensmile_out and openface_out_A\n')
    f.write('Confusion Matrix:\n')
    f.write(str(cm_rocket_a))
    f.write('\n')
    f.write('Classification Report:\n')
    f.write(cr_rocket_a)
    f.write('\n')
    f.write('Rocket on opensmile_A_out and merged_openface_out\n')
    f.write('Confusion Matrix:\n')
    f.write(str(cm_rocket_v))
    f.write('\n')
    f.write('Classification Report:\n')
    f.write(cr_rocket_v)
    f.write('\n')
    f.write('Rocket on opensmile_A_out and openface_out_A\n')
    f.write('Confusion Matrix:\n')
    f.write(str(cm_rocket_va))
    f.write('\n')
    f.write('Classification Report:\n')
    f.write(cr_rocket_va)
    f.write('\n')

    f.write('Rocket on merged_opensmile_out and merged_openface_out with SAX\n')
    f.write('Confusion Matrix:\n')
    f.write(str(cm_rocket_sax))
    f.write('\n')
    f.write('Classification Report:\n')
    f.write(cr_rocket_sax)
    f.write('\n')
    f.write('Rocket on merged_opensmile_out and openface_out_A with SAX\n')
    f.write('Confusion Matrix:\n')
    f.write(str(cm_rocket_a_sax))
    f.write('\n')
    f.write('Classification Report:\n')
    f.write(cr_rocket_a_sax)
    f.write('\n')
    f.write('Rocket on opensmile_A_out and merged_openface_out with SAX\n')
    f.write('Confusion Matrix:\n')
    f.write(str(cm_rocket_v_sax))
    f.write('\n')
    f.write('Classification Report:\n')
    f.write(cr_rocket_v_sax)
    f.write('\n')
    f.write('Rocket on opensmile_A_out and openface_out_A with SAX\n')
    f.write('Confusion Matrix:\n')
    f.write(str(cm_rocket_va_sax))
    f.write('\n')
    f.write('Classification Report:\n')
    f.write(cr_rocket_va_sax)
    f.write('\n')




