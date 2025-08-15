#  Initial flow to treat recordings
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import pandas as pd
import math
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.decomposition import FastICA  # pip install scikit-learn
from sklearn.decomposition import PCA
import warnings
from scipy.signal import welch
import time
import seaborn as sns  # pip install seaborn
from sklearn.metrics import confusion_matrix

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# --------------------------------------- Global Parameters -----------------------------------------------------
#choose model and patient


patient_num = 1
patientData = rf"C:\Users\safra\Documents\לימודים\פרויקט קיץ 2025\DeepLearningProject-master-ido\Recordings\patient{patient_num}_ConvertedData_noTables.mat"
result_str = "patient"+str(patient_num)+"_Wl_"  # for the main: text to add in the saved files

use_saved_model = False
load_path = " .pth"  # "patient1_optimization_second_trymodel.pth"
# for evaluation: update load_path and the hyper parameters of the model below!

save_new_model = True
save_path = result_str+"model.pth"

# Hyper parameters
#MODEL architecture
epoches = 60  # Recommended: 30 for small datasets. for patients 4/5: 100 # for 2 vowels: 40
num_lstm_layers = 1
batch_size = 9  # for patients 4/5: 10
learning_rate = 0.045  # default 0.045
hidden_state_size = 7  # Hidden state size for the LSTM
dropout_flag = True
p_dropout = 0.1 # chance to zero the neuron
pos_weight = torch.tensor([4.0])  # Adding extra weight to positive examples, to compensate for the 40_non_U vs 10_U
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # was: nn.CrossEntropyLoss(), but it doesnt work with one-hot encoded labels.
n_channels = 2
n_vowels = 5  # hard coded
possible_classifications = ['None', 'A', 'E', 'I', 'O', 'U']
input_tensor_size = n_channels  # Number of input features
output_size = len(possible_classifications)  # Number of classes for classification


#training and testing
UnBalanced_train = False  # include all train data
UnBalanced_test = False  # include all train data
training_set_perc = 0.7  # percent of the data that used for training


#preproccesing
ICA_to_all_trials = True
ICA_filter_comp = 0.7
normalize_flag = True
components = [[59, 73, 85, 99], [4, 21, 36, 79, 94], [18, 36, 74, 54, 62], [87, 127], [192], [0, 40, 68, 75, 85]]
noise_components = components[patient_num-1]  # start from 0=patient1
# noise_components = [59, 73, 85, 99]  # for patient2. 45, 32, not included cause of not clean edges
# noise_components = [4, 21, 36, 79, 94]  # for patient1. 69, not included
# noise_components = [18, 36, 74, 54, 62]  # for patient3.
#noise_components = [87, 127]  # for patient4.
# noise_components = [192]  # for patient5.
# noise_components = [0, 40, 68, 75, 85]  # for patient6

#choose channels
Channel2_only = False
Channel1_only = False
if Channel2_only or Channel1_only:
    ICA_to_all_trials = False


#vowel balance
vowels_toRun = ['A', 'E', 'I', 'O', 'U']  # ['A', 'E', 'I', 'O', 'U']
remove_vowels_from_test_only = False  # if True: training on all vowels
# for 2 vowels only: 20-40 epoches, 3-5 BS, 5-6 hidden, 0.8 ICA.
weaken_vowels = True
vowels_to_weaken = ['A']
recs_num_to_weaken = 3  # integer between 1 and 4


# Archived preprocessing:
aggregate_to_bins_flag = False
# bin_size = 3  # how many time points will be averaged
PCA_flag = False
# PCA_filter_first_comp = 0.3  #
ICA_flag = False
# comp_to_reduce = 1  # ICA_flag: 1 for patient2, 0 for patient1
# ICA_filter_comp = 0.3  # ICA_flag :0.3 for patient2, 0.7 for patient1, 0.5 is default


data = scipy.io.loadmat(patientData)
'''
data contains 2 fields - ImageryCellArray, VowelsCellArray.
Each is a table 5X3, where the first column is an array of the first electrode, the second column is second electrode,
 and third column is the label (A/Imagery_U...).
Array of electrode contains: LAST column is timing (x axis), other cols contains ~10 trials, each trial is a column.
 values in these cols are LFP= local field potentials.
'''


# ------------------------------------------ Functions -----------------------------------------------------------


def optimizeLoop(patient_num): #loops through possible hyperparameters and finds the best ones
    result_str = "Patient"+str(patient_num)
    save_path = result_str+"best.pth"
    best_accuracy = 0
    parameters = [0,0,0,0,0,0]
    all_results_df = pd.DataFrame()

# 144 combinations
    ICA_filter_comp_values = [0.3, 0.5, 0.7]  # 1= no ICA filtering
    hidden_state_size_values = [5, 10, 16]
    learning_rate_values = [0.025, 0.035, 0.045, 0.055]  # default 0.045
    num_lstm_layers_values = [1]

    batch_size_options = [[8, 9],[8, 9],[5, 8, 9],[8, 9, 10],[8, 10],[5, 8]]
    batch_size_values = batch_size_options[patient_num-1]  # for patients 4/5: 10

    epoch_options = [[70, 60],[60, 70],[50, 60],[80, 100],[80, 100],[50, 60]]
    epoches_values = epoch_options[patient_num-1]

    dict_patient1 = create_patient_dict(data['VowelsCellArray'])

    for ICA_filter_comp in ICA_filter_comp_values:
        if ICA_to_all_trials:
            noise_components = components[patient_num - 1]
            dict_patient1 = apply_ICA_allTrials(dict_patient1, noise_components, ICA_filter_comp)
            ICA_flag = False  # to make sure not ICA twice

        # set up training set and test set
        train_dataset = patient_TrainingDataset(dict_patient1)
        test_set = patient_Testset(dict_patient1)

        for batch_size in batch_size_values:
            # set up data loader- has no memory so can be done once for each batch size
            sampler = StratifiedBatchSampler(train_dataset, batch_size=batch_size,
                                             num_labels=len(possible_classifications))  # 1+5= num of vowels in the dataset
            dataloader = DataLoader(dataset=train_dataset, sampler=sampler, batch_size=1)  # , num_workers=1)

            num_of_training_trials = len(train_dataset)
            n_iterations = math.ceil(num_of_training_trials / batch_size)  # iterations per epoch

            for num_lstm_layers in num_lstm_layers_values:
                for hidden_state_size in hidden_state_size_values:
                    for learning_rate in learning_rate_values:
                        for epoches in epoches_values:
                            # Set up the LSTM
                            lstm = LSTM(input_tensor_size, hidden_state_size, output_size, batch_size, num_lstm_layers)
                            optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

                            all_losses = []
                            lstm.train()

                            for epoch in range(epoches):
                                epoch_loss = 0
                                for i_batch, (inputs, labels) in enumerate(dataloader):  # this will run n_iterations times every epoch
                                    inputs = inputs.squeeze(0)  # remove dim 0 which len is 1, cause of sampler+dataloader
                                    labels = labels.squeeze(0)

                                    actual_batch_size = inputs.shape[0]  # for the case of residual batch if 35 is not divisible by batch size
                                    hidden = lstm.init_hidden(batch_size=actual_batch_size)  # Initialize the hidden state for the current batch
                                    optimizer.zero_grad()  # Zero the gradients

                                    outputs, hidden = lstm(inputs, hidden)

                                    loss = criterion(outputs, labels[:, 0, :])
                                    loss.backward()  # Backward pass
                                    optimizer.step()  # Update the weights
                                    epoch_loss += loss.item()  # Accumulate the loss for reporting

                                # Track average loss for the epoch
                                avg_epoch_loss = epoch_loss / n_iterations
                                all_losses.append(avg_epoch_loss)
                                print(f"Epoch [{epoch + 1}/{epoches}] completed with Average Loss: {avg_epoch_loss:.4f}")

                            print("Training completed")

                            current_accuracy, current_results_df, conf_mat, label_order = calculate_accuracy_lstm(lstm, test_set)   # TODO- treat conf_mat?
                            if current_accuracy >= best_accuracy:
                                best_accuracy = current_accuracy
                                parameters = [epoches, num_lstm_layers, batch_size, learning_rate, hidden_state_size, ICA_filter_comp]
                                params = {'Trial Tested': "epoches_"+str(epoches)+"_numLayers_"+str(num_lstm_layers),
                                          'Actual Label': "batch_size_"+str(batch_size)+"_LR_"+str(learning_rate),
                                          'Predicted Label': "hidden_"+str(hidden_state_size)+"_ICA_"+str(ICA_filter_comp)+"accuracy:"+str(current_accuracy)+"%"}
                                current_results_df.loc[len(current_results_df)] = params  # Works in all versions

                                all_results_df = pd.concat([all_results_df, current_results_df], axis=1)
                                save_model(lstm, optimizer, "epo=" + str(parameters[0]) + "_nLayer=" + str(parameters[1]) +"_BS=" + str(parameters[2])\
                 +"_lr=" + str(parameters[3]) +"_hid=" + str(parameters[4]) +"_ICA=" + str(parameters[5])+save_path)
                        time.sleep(2)
    # parameters to excel name
    print(parameters)
    excel_file = result_str + "_epo=" + str(parameters[0]) + "_nLayer=" + str(parameters[1]) +"_BS=" + str(parameters[2])\
                 +"_lr=" + str(parameters[3]) +"_hid=" + str(parameters[4]) +"_ICA=" + str(parameters[5]) + ".xlsx"
    all_results_df.to_excel(excel_file, index=False)
    return


def time_vec_unpack(time_unfiltered, time_list):
    for i in range(len(time_unfiltered)):
        item = time_unfiltered[i][0]
        if isinstance(item, str):  # If item is a string
            number = float(item.strip().replace(' sec', ''))
        else:  # If it's already a number
            number = float(item)
        time_list.append(number)


def LFP_unpack(unfiltered_list, new_list):
    for i in range(len(unfiltered_list)):
        new_list.append(unfiltered_list[i][0][0])


def deterministic_mixes(x, y1, y2, seed=42):
    np.random.seed(seed)  # Ensures reproducibility
    mixes = np.array([np.random.choice(range(int(y1), int(y2) + 1), int(x), replace=False) for _ in range(9)])
    return mixes


def create_patient_dict(VowelsCellArray, run_i=0):
    ''' input: VowelsCellArray is 5X3 array as described above, can be production or imaginary
    output: patient dictionary, each dictionary is a labeled input for the NN:
     patient_dict[trial_number] -> keys: LFP_vec_CH{x}, time_vec, label, training_flag
      sub- dictionary for every trial, each patient is expected to have ~50 trials
     '''

    labeling = VowelsCellArray[:, 2]  # ground truth
    patient_dict = {}
    trial_global_number = 0

    vowels_trials_len = []
    for vowel_j in range(len(labeling)):
        vowels_trials_len.append(np.size(VowelsCellArray[:, 0][vowel_j], 1)-1)  # -1 cause of time vector in the end
    minimum_trials_for_vowel = np.min(vowels_trials_len)

    training_set_size_for_vowel = np.round(training_set_perc * minimum_trials_for_vowel)
    test_set_size_for_vowel = minimum_trials_for_vowel - training_set_size_for_vowel

    for vowel_i in range(len(labeling)):
        vowel_trials_data = []  # each electrode
        for i in range(n_channels):  # i will go 0,1
                vowel_trials_data.append(VowelsCellArray[:, i][vowel_i])
                if i >= 1:
                    # expected: num_trials_for_vowel in the columns dim. must be the same in both electrodes.
                    if np.size(vowel_trials_data[i-1], 1) != np.size(vowel_trials_data[i], 1):
                        print('Error! num of trials in all channels needs to be the same')
                        return 0
        vowel = labeling[vowel_i][0][0][0]  # extract the vowel char

        key_time = f"time_vec"
        time_vec_packed = vowel_trials_data[0][:, -1]
        time_vec = []
        time_vec_unpack(time_vec_packed, time_vec)

        num_of_trials = np.size(vowel_trials_data[0], 1) - 1  # -1 cause of the time vector there

        if UnBalanced_train:
            training_set_size_for_vowel = np.round(training_set_perc * num_of_trials)
        if UnBalanced_test:
            test_set_size_for_vowel = num_of_trials - training_set_size_for_vowel  # more tests if more trials per vowel

        training_set_accum = 0
        test_set_accum = 0
        indices_mix = deterministic_mixes(test_set_size_for_vowel, 0, num_of_trials-1)  # assuming 9 models need mixing. num_of_trials-1 cause this is the max index

        for trial_num in range(num_of_trials):  # running on all trial for this vowel
            trial_dict = {}

            if run_i == 0:
                if training_set_accum < training_set_size_for_vowel:
                    trial_dict["training_flag"] = 1
                    training_set_accum = training_set_accum + 1
                else:
                    if test_set_accum < test_set_size_for_vowel:
                        trial_dict["training_flag"] = 0
                        test_set_accum = test_set_accum + 1
                    else:
                        continue  # go to the next trial num, in this case- end loop
            else:
                test_indices = indices_mix[run_i-1]  # start from index 0 for run_i=1. run_i reaches 9
                if trial_num in test_indices:
                    trial_dict["training_flag"] = 0
                    test_set_accum = test_set_accum + 1  # for debug only
                else:
                    if training_set_accum < training_set_size_for_vowel:
                        trial_dict["training_flag"] = 1
                        training_set_accum = training_set_accum + 1
                    else:
                        continue  # go to the next trial num, maybe it is test

            trial_dict[key_time] = time_vec
            for num in range(1, n_channels + 1):
                key_LFP = f"LFP_vec_CH{num}"
                trial_data_packed = vowel_trials_data[:][num - 1][:, trial_num]

                if Channel2_only:
                    trial_data_packed = vowel_trials_data[:][1][:, trial_num]  # num - 1 -> 1 means ch1 data will be in channel 2 too
                if Channel1_only:
                    trial_data_packed = vowel_trials_data[:][0][:,
                                        trial_num]  # num - 1 -> 0 means ch1 data will be in channel 1 too


                LFP_vec = []
                LFP_unpack(trial_data_packed, LFP_vec)

                #  preprocessing:
                if normalize_flag:
                    LFP_vec = preprocessing_normalize_vec(LFP_vec)
                # if aggregate_to_bins_flag:
                #     LFP_vec = preprocessing_bins_aggregation_vec(LFP_vec, bin_size)

                trial_dict[key_LFP] = LFP_vec

            # if PCA_flag:
            #     PCA_vec = apply_PCA(trial_dict["LFP_vec_CH1"], trial_dict["LFP_vec_CH2"])
            #     trial_dict["PCA_vec"] = PCA_vec
            #
            # if ICA_flag:
            #     ICA_combined_vec = apply_ICA(trial_dict["LFP_vec_CH1"], trial_dict["LFP_vec_CH2"])
            #     trial_dict["ICA_vec"] = ICA_combined_vec

            trial_dict["label"] = vowel

            patient_dict[trial_global_number] = trial_dict
            trial_global_number += 1

    return patient_dict


def plot_error_graph(patient_dict, labels_to_plot):
    """
    Plots an error graph for the given patient dictionary.

    Parameters:
    - patient_dict: Dictionary where each key corresponds to a trial. Each trial has:
        - LFP_vec_CH1, LFP_vec_CH2, ..., LFP_vec_CHn: Voltage vectors for each channel
        - time_vec: Time vector (not used here)
        - label: Label for the trial
    - labels_to_plot: List of labels to include in the plot.

    Output:
    - Error graph with min, max, mean, and standard deviation (std) for each label/trial.
    """
    # Prepare data for plotting
    x_positions = []
    means = []
    mins = []
    maxs = []
    std_devs = []
    labels = []  # Labels corresponding to each x-position

    # Process each trial in patient_dict
    for trial_key, trial_data in patient_dict.items():
        label = trial_data["label"]
        if label in labels_to_plot:
            # Combine all LFP vectors for this trial
            all_voltages = []
            for key in trial_data:
                if key.startswith("LFP_vec_CH"):
                    all_voltages.extend(trial_data[key])

            # Compute statistics
            means.append(np.mean(all_voltages))
            mins.append(np.min(all_voltages))
            maxs.append(np.max(all_voltages))
            std_devs.append(np.std(all_voltages))  # Compute standard deviation

            # Assign a unique x-position and store the label
            x_positions.append(len(x_positions))  # Unique position for each trial
            labels.append(label)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot min-max range as a vertical line
    for i in range(len(x_positions)):
        plt.plot([x_positions[i], x_positions[i]], [mins[i], maxs[i]], color='gray', alpha=0.7)

    # Plot mean as a bold point with error bars (standard deviation)
    plt.errorbar(x_positions, means, yerr=std_devs, fmt='o', color='blue',
                 markersize=8, capsize=5, label="Mean ± Std Dev")

    # Customize x-axis
    plt.xticks(ticks=x_positions, labels=labels, rotation=45)
    plt.xlabel("Labels (Trial Groups)")
    plt.ylabel("Voltage")
    plt.title("Error Graph of Voltage Vectors with Mean, Min-Max, and Std Dev")
    plt.grid(alpha=0.3)
    plt.legend(["Min-Max Range", "Mean ± Std Dev"], loc="upper left")
    plt.tight_layout()
    plt.show()


def get_classification(output):   # made changes for lstm !!!
    print(output)
    # Flatten the output tensor if necessary (it should be [1, 6])
    output = output.squeeze(0)  # Now output is [6]
    classification_index = torch.argmax(output)

    # If the highest score corresponds to the first class (index 0),
    # find the maximum index in the remaining classes (ignoring the first class)
    if classification_index == 0:
        classification_index = 1 + torch.argmax(output[1:])

    # Use the index to get the corresponding classification
    classification = possible_classifications[classification_index]

    return classification
    # classification = possible_classifications[torch.argmax(output)]
    # # classification = possible_classifications[torch.argmin(output)]
    # if torch.argmax(output) == 0:  # if None and A are both highest, it is an A
    #     classification = possible_classifications[1+torch.argmax(output[1:])]
    # return classification  # ideal output will be a vector with 1 index, like: 100010


def calculate_accuracy_lstm(lstm, dataset, save_plot=False, save_confMat_excel=False, save_str=""):
    # Calculate the accuracy of the model on a given dataset
    correct = 0
    total = 0
    trials_tested = []
    ground_truth = []
    predicted_labels = []
    lstm.eval()
    if save_str=="":
        save_str = "patient_" + str(patient_num)

    for i in range(len(dataset)):
        # TODO- add trial num
        input_tensor, label = dataset[i]
        input_tensor = input_tensor.unsqueeze(0)
        label = label.unsqueeze(0)

        hidden = lstm.init_hidden(batch_size=1)  # Initialize hidden state
        output, hidden = lstm(input_tensor, hidden)

        trials_tested.append(i)

        predicted_label = get_classification(output)  # Model's predicted classification
        predicted_labels.append(predicted_label)

        true_label = get_classification(label[0, 0])  # The true label
        ground_truth.append(true_label)

        if predicted_label == true_label:  # Check if the prediction matches the true label
            correct += 1
        total += 1

    # Create results dataframe
    results_df = pd.DataFrame({
        "Trial Tested": trials_tested,
        "Actual Label": ground_truth,
        "Predicted Label": predicted_labels
    })

    # Calculate accuracy
    accuracy = (correct / total) * 100
    print(str(len(dataset)) + f" Accuracy: {accuracy:.2f}% ({correct}/{total})")

    # Define fixed label order (sorted unique labels)
    label_order = sorted(np.unique(ground_truth))  # Ensures the same order across runs

    # Compute confusion matrix with fixed label order
    cm = confusion_matrix(ground_truth, predicted_labels, labels=label_order)
    cm = (cm / cm.sum(axis=1, keepdims=True)) * 100
    cm = np.nan_to_num(cm)

    if save_plot:
        # Plot confusion matrix
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", xticklabels=label_order, yticklabels=label_order)
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.title("Confusion Matrix", fontweight='bold', fontsize=13)
        plt.savefig(save_str+"_conf_matrix.png", dpi=300, bbox_inches='tight')
        print("Confusion matrix plot saved as .png")
        plt.show()

    if save_confMat_excel:  # Save confusion matrix to Excel if needed
        cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)
        cm_df.to_excel(save_str + "_conf_matrix.xlsx", index=True)
        print("Confusion matrix saved as xlsx")


    return accuracy, results_df, cm, label_order


def save_model(model, optimizer, save_path):
    # Save the model and optimizer state to a file
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)
    print(f"Model and optimizer saved to {save_path}")


def load_model(model, optimizer, load_path):
    # Load the model and optimizer state from a file
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model and optimizer loaded from {load_path}")


def preprocessing_normalize_vec(LFP_vec):
    normalized_vec = LFP_vec-np.mean(LFP_vec)
    normalized_vec = normalized_vec/(np.max(normalized_vec)-np.min(normalized_vec))
    return normalized_vec


def preprocessing_bins_aggregation_vec(LFP_vec, bin_size):
    LFP_vec = np.array(LFP_vec)
    vec_reshaped = LFP_vec[:len(LFP_vec) // bin_size * bin_size].reshape(-1, bin_size)
    aggregated_vec = np.mean(vec_reshaped, axis=1)  # Calculate the mean of each bin
    if len(LFP_vec) % bin_size != 0:  # Handle leftovers
        leftover_mean = np.mean(LFP_vec[len(LFP_vec) // bin_size * bin_size:])
        aggregated_vec = np.append(aggregated_vec, leftover_mean)
    return aggregated_vec


def apply_ICA_allTrials(patient_dict, noise_components, ICA_filter_comp):
    n_trials = len(patient_dict)

    for trial_key, trial_data in patient_dict.items():
        LFP_vec_CH1 = trial_data["LFP_vec_CH1"]
        LFP_vec_CH2 = trial_data["LFP_vec_CH2"]
        if trial_key == 0:
            lfp_all_trials = np.stack([LFP_vec_CH1, LFP_vec_CH2], axis=1)  # Shape: (8000, 2n_trials)
        else:
            LFP_vec_CH1 = np.expand_dims(LFP_vec_CH1, axis=1)
            LFP_vec_CH2 = np.expand_dims(LFP_vec_CH2, axis=1)
            lfp_all_trials = np.concatenate([lfp_all_trials, LFP_vec_CH1, LFP_vec_CH2], axis=1)  # Shape: (8000, 2n_trials)

    # Apply ICA to extract independent sources
    ica = FastICA(n_components=n_trials*2, random_state=42, max_iter=500, tol=1e-3)
    lfp_ica = ica.fit_transform(lfp_all_trials)  # output Shape: (8000, n_trials*2)

    # fs = 2000  # 8000/4 sec
    # ica_components = lfp_ica.T
    # num_components = ica_components.shape[0]
    # plt.figure(figsize=(12, 6))
    # for i in range(num_components):
    #     f, Pxx = welch(ica_components[i], fs=fs, nperseg=1024)
    #     plt.semilogy(f, Pxx, label=f'Component {i}')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power Spectral Density (dB/Hz)')
    # plt.title('PSD of ICA Components')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # ica_sources = lfp_ica
    # num_components = ica_sources.shape[1]
    # components_per_figure = 10  # Change this number based on preference
    # for i in range(0, num_components, components_per_figure):
    #     fig, axes = plt.subplots(components_per_figure, 1, figsize=(10, 8), sharex=True)
    #
    #     for j in range(components_per_figure):
    #         if i + j < num_components:
    #             axes[j].plot(ica_sources[:, i + j])
    #             axes[j].set_title(f"ICA Component {i + j}")
    #
    #     axes[-1].set_xlabel("Time (samples)")
    #     plt.tight_layout()
    #     plt.show()


    lfp_cleaned = lfp_ica
    for num in noise_components:  # If needed, reduce unwanted components and inverse transform
        lfp_cleaned[:, num] = lfp_ica[:, num] * ICA_filter_comp

    lfp_cleaned_signals = ica.inverse_transform(lfp_cleaned)

    trial_index = 0
    for trial_key, trial_data in patient_dict.items():
        trial_data["ICA_vec"] = lfp_cleaned_signals[:, 2*trial_index:2*trial_index+2]
        trial_data["ICA_vec"] = (trial_data["ICA_vec"] - np.mean(trial_data["ICA_vec"], axis=0))\
                                /np.std(trial_data["ICA_vec"], axis=0)  # normalize the new signal

        trial_index = trial_index+1

        trial_data["ICA_vec"] = preprocessing_normalize_vec(trial_data["ICA_vec"])

        # # Plot before and after ICA
        # plt.figure(figsize=(10, 5))
        # plt.plot(trial_data["LFP_vec_CH1"], label="Noisy Signal", alpha=0.5)
        # plt.plot(trial_data["LFP_vec_CH2"], label="Noisy Signal2", alpha=0.5)
        # plt.plot(trial_data["ICA_vec"][:, 0], label="Cleaned Signal (ICA) CH1", linewidth=1)
        # plt.plot(trial_data["ICA_vec"][:, 1], label="Cleaned Signal (ICA) CH2", linewidth=1)
        # plt.legend()
        # plt.xlabel("Time Points")
        # plt.ylabel("Amplitude")
        # plt.title("ICA Artifact Removal in LFP Signals")
        # plt.show()

    return patient_dict

# ---- function for LSTM ------

def run_LSTM_on_trial_dict(lstm, trial_obj):
    # Initialize hidden and cell states
    hidden = lstm.init_hidden(batch_size=1)  # (h_0, c_0)

    # channels, label = trial_objects
    # channels = torch.stack([item["feature"] for item in trial_objects])  # Shape: (batch_size, feature_dim)
    # labels = torch.tensor([item["label"] for item in trial_objects])
    # Assuming all channels have the same length
    channels, label = trial_obj
    input_vector_size = channels.shape[0]  # 8000

    for i in range(input_vector_size):
        input_tensor = channels[i].unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, input_size]
        output, hidden = lstm(input_tensor, hidden)

    return output


def train_lstm(lstm, trial_LFPs, ideal_output_tensors, optimizer):  # each iteration here changes th NN
    lstm.train()

    #TODO- everything here is obsolete

    hidden = lstm.init_hidden(batch_size=batch_size)

    # for i in range(input_vector_size):
    input_tensor = trial_LFPs.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, input_size]
    output, hidden = lstm(input_tensor, hidden)  # hidden = (h,c)

    outputs = torch.empty([0, 6])
    for i in batch_size:
        output = run_LSTM_on_trial_dict(lstm, trial_LFPs)
        outputs = torch.cat((outputs, output[None, ...]), dim=0)

    # Debugging: Check the shapes of output and ideal_output_tensor
    print(f"Output shape: {outputs.shape}")
    print(f"Ideal output shape: {ideal_output_tensors.shape}")

    loss = criterion(outputs, ideal_output_tensors)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()


def evaluate_model(saved_path, patient_num, hidden_state_size, batch_size, learning_rate, ICA_filter_comp, epoches):
    lstm = LSTM(input_tensor_size, hidden_state_size, output_size, batch_size, num_lstm_layers)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

    load_model(lstm, optimizer, saved_path)
    dict_patient1 = create_patient_dict(data['VowelsCellArray'])
    if ICA_to_all_trials:
        noise_components = components[patient_num - 1]
        dict_patient1 = apply_ICA_allTrials(dict_patient1, noise_components, ICA_filter_comp)

    # set up test set
    test_set = patient_Testset(dict_patient1)
    accuracy, results_df, conf_mat, label_order = calculate_accuracy_lstm(lstm, test_set, save_plot=True, save_confMat_excel=True)  # TODO- treat conf_mat
    parameters = [epoches, num_lstm_layers, batch_size, learning_rate, hidden_state_size, ICA_filter_comp]
    params = {'Trial Tested': "epoches_" + str(epoches) + "_numLayers_" + str(num_lstm_layers),
                  'Actual Label': "batch_size_" + str(batch_size) + "_LR_" + str(learning_rate),
                  'Predicted Label': "hidden_" + str(hidden_state_size) + "_ICA_" + str(
                      ICA_filter_comp) + "accuracy:" + str(accuracy) + "%"}
    results_df.loc[len(results_df)] = params  # Works in all versions

    # parameters to excel name
    excel_file = "Evaluation_Patient"+str(patient_num) + "_epo=" + str(parameters[0]) + "_nLayer=" + str(parameters[1]) + "_BS=" + str(parameters[2]) \
                 + "_lr=" + str(parameters[3]) + "_hid=" + str(parameters[4]) + "_ICA=" + str(parameters[5]) + ".xlsx"
    results_df.to_excel(excel_file, index=False)
    return


def create_n_run_models(epoches, num_lstm_layers, batch_size, learning_rate, hidden_state_size, ICA_filter_comp, patient_num, vowels_toRun=['A', 'E', 'I', 'O', 'U']):
    num_of_runs = 10

    noise_components = components[patient_num - 1]
    accuracies = []
    conf_mats = []
    all_results_df = pd.DataFrame()

    for run_i in range(num_of_runs):
        result_str = "Patient_" + str(patient_num) + "_run_" + str(run_i)

        dict_patient = create_patient_dict(data['VowelsCellArray'], run_i)
        if ICA_to_all_trials:
            dict_patient = apply_ICA_allTrials(dict_patient, noise_components, ICA_filter_comp)

        # Removing vowels if needed
        if vowels_toRun != ['A', 'E', 'I', 'O', 'U']:
            result_str = result_str + "_" + vowels_toRun[0] + vowels_toRun[1]
            for key in list(dict_patient.keys()):  # Convert keys to list
                trial = dict_patient[key]
                if trial["label"] not in vowels_toRun:
                    if not (trial["training_flag"] == 1 and remove_vowels_from_test_only):
                        del dict_patient[key]  # if the record is for testing, or we want to train only on the vowels_to_run- delete.

        # Weaken vowels if needed
        if weaken_vowels:  # this is only in the train set and not in the test
            result_str = result_str + "_Weak_" + vowels_to_weaken[0]
            accum_recs_weak = [0]*len(vowels_to_weaken)
            for key in list(dict_patient.keys()):  # Convert keys to list
                trial = dict_patient[key]
                if (trial["label"] in vowels_to_weaken) and (trial["training_flag"] == 1):
                    ind = vowels_to_weaken.index(trial["label"])
                    if accum_recs_weak[ind] < recs_num_to_weaken:
                        del dict_patient[key]
                        accum_recs_weak[ind] = accum_recs_weak[ind]+1

        special_train_dataset = patient_TrainingDataset(dict_patient)
        special_test_set = patient_Testset(dict_patient)

        sampler = StratifiedBatchSampler(special_train_dataset, batch_size=batch_size, num_labels=len(possible_classifications))
        dataloader = DataLoader(dataset=special_train_dataset, sampler=sampler, batch_size=1)
        num_of_training_trials = len(special_train_dataset)
        n_iterations = math.ceil(num_of_training_trials / batch_size)  # iterations per epoch

        # Set up the LSTM
        lstm = LSTM(input_tensor_size, hidden_state_size, output_size, batch_size, num_lstm_layers)
        optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, betas=(0.9, 0.999),eps=1e-08)
        all_losses = []
        lstm.train()

        for epoch in range(epoches):
            epoch_loss = 0
            for i_batch, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.squeeze(0)  # remove dim 0 which len is 1, cause of sampler+dataloader
                labels = labels.squeeze(0)
                actual_batch_size = inputs.shape[0]  # for the case of residual batch if 35 is not divisible by batch size
                hidden = lstm.init_hidden(batch_size=actual_batch_size)  # Initialize the hidden state for the current batch
                optimizer.zero_grad()  # Zero the gradients

                outputs, hidden = lstm(inputs, hidden)

                loss = criterion(outputs, labels[:, 0, :])
                loss.backward()  # Backward pass
                optimizer.step()  # Update the weights
                epoch_loss += loss.item()  # Accumulate the loss for reporting

            avg_epoch_loss = epoch_loss / n_iterations
            all_losses.append(avg_epoch_loss)
            print(f"Epoch [{epoch + 1}/{epoches}] completed with Average Loss: {avg_epoch_loss:.4f}")

        print("Training completed")

        save_model(lstm, optimizer, result_str + "_model.pth")
        current_accuracy, current_results_df, current_conf_mat, label_order = calculate_accuracy_lstm(lstm, special_test_set, save_plot=False, save_confMat_excel=True, save_str=result_str)  # TODO- treat conf_mat

        params = {'Trial Tested': "accuracy:", 'Actual Label': str(current_accuracy), 'Predicted Label': "%"}
        current_results_df.loc[len(current_results_df)] = params
        all_results_df = pd.concat([all_results_df, current_results_df], axis=1)
        accuracies.append(current_accuracy)
        conf_mats.append(current_conf_mat)
        time.sleep(2)

    avg_accuracy = sum(accuracies)/len(accuracies)
    std_error_accuracy = np.std(accuracies)/np.sqrt(num_of_runs)
    avg_conf_mat = np.mean(conf_mats, axis=0)
    std_error_conf_mat = np.std(conf_mats, axis=0)/np.sqrt(num_of_runs)

    format_string = np.vectorize(lambda x, y: f"{x:.1f}+-{y:.1f}")
    conf_mat_total = format_string(avg_conf_mat, std_error_conf_mat)

    plt.figure(figsize=(5, 4))
    sns.heatmap(avg_conf_mat, annot=True, fmt=".1f", cmap="Blues", xticklabels=label_order, yticklabels=label_order)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix", fontweight='bold', fontsize=13)
    plt.savefig("Average_ConfMat_patient_"+str(patient_num)+"ACCU_"+ "{:.1f}".format(avg_accuracy) +"_std_err_"+"{:.1f}".format(std_error_accuracy)+".png", dpi=300, bbox_inches='tight')
    plt.show()
    cm_df = pd.DataFrame(conf_mat_total, index=label_order, columns=label_order)
    cm_df.to_excel("Average_ConfMat_patient_"+str(patient_num)+"ACCU_"+ "{:.1f}".format(avg_accuracy) +"_std_err_"+"{:.1f}".format(std_error_accuracy)+".xlsx", index=True)

    parameters = [epoches, num_lstm_layers, batch_size, learning_rate, hidden_state_size, ICA_filter_comp]
    excel_file = "Patient_" +str(patient_num)+"_Nmodels_ACCU_" + "{:.1f}".format(avg_accuracy) +"_std_err_"+"{:.1f}".format(std_error_accuracy)+"_epo=" + str(parameters[0]) + "_nLay=" + str(parameters[1]) + "_BS=" + str(parameters[2]) \
                     + "_lr=" + str(parameters[3]) + "_hid=" + str(parameters[4]) + "_ICA=" + str(
            parameters[5]) + ".xlsx"
    all_results_df.to_excel(excel_file, index=False)
    print("avg_accuracy: "+"{:.1f}".format(avg_accuracy)+"%_std_err_"+"{:.1f}".format(std_error_accuracy)+"%")
    return


# -------------------------------------------- Classes -------------------------------------------------------------
class patient_TrainingDataset(Dataset):
    # TODO- add trial num
    def __init__(self, patientDict):
        # Initialize dataset
        all_concatedLFP_of_patient = []  # List to store trials
        all_vowels_list = []  # List to store labels
        for trial_key, trial_data in patientDict.items():
            if trial_data["training_flag"] == 1:
                # Convert LFP data for both channels to tensors
                ch1_tensor = torch.tensor(trial_data["LFP_vec_CH1"], dtype=torch.float32)
                ch2_tensor = torch.tensor(trial_data["LFP_vec_CH2"], dtype=torch.float32)
                concatedLFP = torch.stack([ch1_tensor, ch2_tensor], dim=-1)  # Shape: [8000, 2]
                # so now:  concatedLFP[0]= CH1(t=0),CH2(t=0)

                # if PCA_flag:
                #     concatedLFP = torch.tensor(trial_data["PCA_vec"], dtype=torch.float32) # overrides the origimal LFP vectors with filtered data

                if ICA_flag or ICA_to_all_trials:
                    concatedLFP = torch.tensor(trial_data["ICA_vec"],
                                               dtype=torch.float32)  # overrides the origimal LFP vectors with filtered data

                all_concatedLFP_of_patient.append(concatedLFP)  # Append each trial to the list

                # Generate ideal output for one trial
                ideal_output = torch.zeros(len(possible_classifications), dtype=torch.float32)
                ideal_output[0] = 1  # Always include vowel presence

                ideal_output[possible_classifications.index(trial_data["label"])] = 1  # Set the correct vowel class to 1

                # Expand the ideal_output to match the sequence length
                sequence_length = 8000  # TODO- should not be hard-coded
                sequence_labels = ideal_output.repeat(sequence_length, 1)  # Shape: [sequence_length, features_out]

                all_vowels_list.append(sequence_labels)  # Add to the list of labels

        # Stack trials and labels into tensors
        self.x_data = torch.stack(all_concatedLFP_of_patient, dim=0)  # Shape: [n_trials, 8000, 2]
        self.y_data = torch.stack(all_vowels_list, dim=0)  # Shape: [n_trials, 6]
        self.n_samples = self.x_data.shape[0]  # Number of trials in dataset

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]  # gives one trial data and vowel

    # we can call len(dataset) to return the size, num of trials
    def __len__(self):
        return self.n_samples


class patient_Testset(Dataset):
    # TODO- add trial num
    def __init__(self, patientDict):
        # Initialize dataset
        all_concatedLFP_of_patient = []  # List to store trials
        all_vowels_list = []  # List to store labels
        for trial_key, trial_data in patientDict.items():
            if trial_data["training_flag"] == 0:  # the only difference from the training set
                # Convert LFP data for both channels to tensors
                ch1_tensor = torch.tensor(trial_data["LFP_vec_CH1"], dtype=torch.float32)
                ch2_tensor = torch.tensor(trial_data["LFP_vec_CH2"], dtype=torch.float32)
                concatedLFP = torch.stack([ch1_tensor, ch2_tensor], dim=-1)  # Shape: [8000, 2]
                # so now:  concatedLFP[0]= CH1(t=0),CH2(t=0)

                # if PCA_flag:
                #     concatedLFP = torch.tensor(trial_data["PCA_vec"], dtype=torch.float32) # overrides the origimal LFP vectors with filtered data

                if ICA_flag or ICA_to_all_trials:
                    concatedLFP = torch.tensor(trial_data["ICA_vec"],
                                               dtype=torch.float32)  # overrides the origimal LFP vectors with filtered data

                all_concatedLFP_of_patient.append(concatedLFP)  # Append each trial to the list

                # Generate ideal output for one trial
                ideal_output = torch.zeros(len(possible_classifications), dtype=torch.float32)
                ideal_output[0] = 1  # Always include vowel presence
                ideal_output[possible_classifications.index(trial_data["label"])] = 1  # Set the correct vowel class to 1

                # Expand the ideal_output to match the sequence length
                sequence_length = 8000  # TODO- should not be hard-coded
                sequence_labels = ideal_output.repeat(sequence_length, 1)  # Shape: [sequence_length, features_out]

                all_vowels_list.append(sequence_labels)  # Add to the list of labels

        # Stack trials and labels into tensors
        self.x_data = torch.stack(all_concatedLFP_of_patient, dim=0)  # Shape: [n_trials, 8000, 2]
        self.y_data = torch.stack(all_vowels_list, dim=0)  # Shape: [n_trials, 6]
        self.n_samples = self.x_data.shape[0]  # Number of trials in dataset

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]  # gives one trial data and vowel

    # we can call len(dataset) to return the size, num of trials
    def __len__(self):
        return self.n_samples


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, num_lstm_layers):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_lstm_layers = num_lstm_layers
        self.output_size = output_size
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_lstm_layers, batch_first=True)  # TODO: can add here num layers to complex the structure  # if batch_first: dims are (batch, seqeunce, features)
        if dropout_flag:
            self.dropout = nn.Dropout(p=p_dropout)

        self.input2output = nn.Linear(hidden_size, output_size)
        # self.stabilizer = nn.LogSoftmax(dim=1)


    def init_hidden(self, batch_size):
        # Initialize the hidden state (h_0) and cell state (c_0) for LSTM
        h_0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size)  # Hidden state # self.num_lstm_layers,
        c_0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size)  # Cell state: Changes less than the hidden
        return (h_0, c_0)


    def forward(self, input_tensor, hidden):
        raw_output, hidden = self.lstm(input_tensor, hidden)  # Output and updated hidden states
        if dropout_flag:
            raw_output = self.dropout(raw_output)
        output = self.input2output(raw_output[:, -1])  # Use the last output of the sequence. raw_output is [batch size ,sequence_len, hidden size]
        # output = self.stabilizer(output)  # Apply LogSoftmax (optional)
        return output, hidden


class StratifiedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_labels):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_labels = num_labels

        # Assume dataset has 'labels' as a matrix (N x sequence_len x num_labels)
        self.labels = dataset.y_data

        # Collect indices for each label except for the first one (since it's always 1)
        self.label_indices = {i: np.where(self.labels[:, 0, i] == 1)[0] for i in range(1, self.num_labels)}

        # Compute total number of samples
        self.total_samples = len(dataset)


    def __iter__(self):
        # Flatten and shuffle indices for all labels
        all_indices = []
        for i in range(1, self.num_labels):
            all_indices.extend(self.label_indices[i])

        np.random.shuffle(all_indices)

        # Create batches, including the last partial batch
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            yield batch  # Yield partial batch as well, if it exists

    def __len__(self):
        return (self.total_samples + self.batch_size - 1) // self.batch_size

# ----------------------------------------- Main Function---------------------------------------------------------

def main_lstm(data):
    dict_patient1 = create_patient_dict(data['VowelsCellArray'])
    if ICA_to_all_trials:
        print ("ran with ICA to all trials")
        dict_patient1 = apply_ICA_allTrials(dict_patient1, noise_components, ICA_filter_comp)
        ICA_flag = False  # to make sure not ICA twice

    # set up training set and dataloader
    train_dataset = patient_TrainingDataset(dict_patient1)

    all_results_df = pd.DataFrame()
    all_validation_results_df = pd.DataFrame()

    for index in range(1, 2):  # Optimization loop, not used now

        sampler = StratifiedBatchSampler(train_dataset, batch_size=batch_size,
                                         num_labels=len(possible_classifications))  # 1+5= num of vowels in the dataset
        dataloader = DataLoader(dataset=train_dataset, sampler=sampler, batch_size=1)  # , num_workers=1)

        # Set up the LSTM
        lstm = LSTM(input_tensor_size, hidden_state_size, output_size, batch_size, num_lstm_layers)
        optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

        if use_saved_model:
            print ("loading model from "+ load_path)
            load_model(lstm, optimizer, load_path)

        # Batches Training Loop
        num_of_training_trials = len(train_dataset)
        n_iterations = math.ceil(num_of_training_trials/batch_size)  # iterations per epoch

        all_losses = []
        plot_steps = 20

        lstm.train()
        for epoch in range(epoches):
            epoch_loss = 0
            for i_batch, (inputs, labels) in enumerate(dataloader):  # this will run n_iterations times every epoch
                inputs = inputs.squeeze(0)  # remove dim 0 which len is 1, cause of sampler+dataloader
                labels = labels.squeeze(0)

                actual_batch_size = inputs.shape[0]  # for the case of residual batch if 35 is not divisible by batch size
                hidden = lstm.init_hidden(batch_size=actual_batch_size)  # Initialize the hidden state for the current batch
                optimizer.zero_grad()  # Zero the gradients

                outputs, hidden = lstm(inputs, hidden)

                loss = criterion(outputs, labels[:, 0, :])  # does labels need to be  labels[:,0,:] ??
                loss.backward()  # Backward pass
                optimizer.step()  # Update the weights

                epoch_loss += loss.item()  # Accumulate the loss for reporting

                if (i_batch + 1) % plot_steps == 0:  # Print loss every `plot_steps` batches
                    print(
                        f"Epoch [{epoch + 1}/{epoches}], Step [{i_batch + 1}/{n_iterations}], Loss: {loss.item():.4f}")

            # Track average loss for the epoch
            avg_epoch_loss = epoch_loss / n_iterations
            all_losses.append(avg_epoch_loss)
            print(f"Epoch [{epoch + 1}/{epoches}] completed with Average Loss: {avg_epoch_loss:.4f}")

        print("Training completed")

        # Plot the training loss
        plt.figure()
        plt.plot(all_losses)
        plt.title("Training Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.show()

    # ------------------------------------------------------Testing---------------------------------------------------
        test_set = patient_Testset(dict_patient1)
        accuracy, results_df, conf_mat, label_order = calculate_accuracy_lstm(lstm, test_set) # TODO- treat conf_mat?
        all_results_df = pd.concat([all_results_df, results_df], axis=1)

        validation_accuracy, results_df_valid, conf_mat, label_order = calculate_accuracy_lstm(lstm, train_dataset) # TODO- treat conf_mat?
        all_validation_results_df = pd.concat([all_validation_results_df, results_df_valid], axis=1)

    excel_file = result_str+"Test_lstm_BS=" + str(batch_size) + "_lr=" + str(learning_rate) + ".xlsx"
    excel_file_val = result_str+"Validate_lstm_BS=" + str(batch_size) + "_lr=" + str(learning_rate) + ".xlsx"
    all_results_df.to_excel(excel_file, index=False)
    all_validation_results_df.to_excel(excel_file_val, index=False)


    if save_new_model:
        save_model(lstm, optimizer, save_path)

    print(results_df)

# ----------------------------------------------main----------------------------------------------------------

# main_lstm(data)
# optimizeLoop(patient_num)
# evaluate_model(load_path, patient_num, hidden_state_size, batch_size, learning_rate, ICA_filter_comp, epoches)
create_n_run_models(epoches, num_lstm_layers, batch_size, learning_rate, hidden_state_size, ICA_filter_comp, patient_num, vowels_toRun)

# dict_patient1 = create_patient_dict(data['VowelsCellArray'])
# plot_error_graph(dict_patient1, possible_classifications)

