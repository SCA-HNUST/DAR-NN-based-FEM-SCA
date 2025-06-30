import os.path
import sys
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import random
from tqdm import tqdm
import os


AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

def load_sca_model(model_file):
    try:
        model = load_model(model_file)
    except:
        print("Error: can't load Keras model file '%s'" % model_file)
        sys.exit(-1)
    return model
    
def get_prediction(model, Traces):
    input_layer_shape = model.input_shape

    # Sanity check
    if input_layer_shape[1] != len(Traces[0]):
        print("Error: model input shape %d instead of %d is not expected ..." % (
            input_layer_shape[1], len(Traces[0])))
        sys.exit(-1)

    
    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        # This is a MLP
        input_data = Traces
    elif len(input_layer_shape) == 3:
        # This is a CNN: reshape the data
        input_data = Traces
        input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        sys.exit(-1)

    # Predict our probabilities
    predictions = model.predict(input_data)
    return predictions
    
def prediction_to_probability(selected_cts_interest, selected_predictions, NUMBER):
    probabilities_array = []

    for i in range(NUMBER):

        probabilities = np.zeros(256)

        for j in range(256):
            
            sbox_out= selected_cts_interest[i]^j
            
            value = sbox_out
           		
            probabilities[j] = selected_predictions[i][value]

        probabilities_array.append(probabilities)

    probabilities_array = np.array(probabilities_array)


    for i in range(len(probabilities_array)):
        if np.count_nonzero(probabilities_array[i]) != 256:
            none_zero_predictions = [a for a in probabilities_array[i] if a != 0]

            min_v = min(none_zero_predictions)

            probabilities_array[i] = probabilities_array[i] + min_v**2	

    return probabilities_array
    
 def rank_cal(selected_probabilities, key_interest, NUMBER):
    rank = []

    total_pro = np.zeros(256)

    for i in range(NUMBER):

        total_pro += np.log(selected_probabilities[i] + 1e-12)  # add a very samll value to avoid log(0)

        sorted_proba = np.array(list(map(lambda a: total_pro[a], total_pro.argsort()[::-1])))

        real_key_rank = np.where(sorted_proba == total_pro[key_interest])[0][0]

        rank.append(real_key_rank)

    rank = np.array(rank)

    return rank
    
def test(model_path, Traces, key_interest, cts_interest, num_trace):
    model = load_sca_model(model_path)
    predictions = get_prediction(model, Traces)
    average = 50
    ranks_array = []

    for i in range(average):
        select = random.sample(range(len(Traces)), num_trace)
        selected_cts_interest = cts_interest[select]
        selected_predictions = predictions[select]

        probabilities = prediction_to_probability(selected_cts_interest, selected_predictions, num_trace)
        ranks = rank_cal(probabilities, key_interest, num_trace)
        ranks_array.append(ranks)

    ranks_array = np.array(ranks_array)
    #print(np.count_nonzero(ranks_array))

    for PGE in range(ranks_array.shape[1]):
        if np.count_nonzero(ranks_array[:, PGE]) < int(average/2):
            
            #print(PGE)
            break

    average_ranks = np.sum(ranks_array, axis=0) / average
    #print(average_ranks)
    #plt.plot(average_ranks)
    #plt.xlabel('Number of Traces')
    #plt.ylabel('PGE')
    #plt.show()
    #plt.close()

    return PGE
    
def load_data(num_avg, data_path, interest_byte=0, shuffle=True):
    
    trace_name = data_path + '/' + str(num_avg) + '_nor_traces_maxmin.npy'
    key_name = data_path + '/' + str(num_avg) + '_10th_roundkey.npy'
    ct_name = data_path + '/' + str(num_avg) + '_ct.npy'
    

    number_total_trace = 5000

    # Load the testing data
    all_traces = np.load(trace_name)
    all_keys = np.load(key_name)
    all_cts = np.load(ct_name)

    if shuffle:
        permutation = np.random.RandomState(seed=42).permutation(all_traces.shape[0])
    else:
        permutation = np.arange(all_traces.shape[0])

    # Randomly select data for the test
    selected_indices = permutation[:number_total_trace]
    print(selected_indices)
    Traces = all_traces[selected_indices, :]
    Traces = Traces[:, 200:220]     #Points of Interest
    key = all_keys
    cts = all_cts[selected_indices]

    key_interest = key[interest_byte]
    cts_interest = cts[:, interest_byte]

    return Traces, key_interest, cts_interest
    
if __name__ == "__main__":
    num_avg = 1
    models_folder = 'DAR-NN_model'
    file = 'DAR-NN.h5'
    num_trace = 3000
    interest_byte = 0
    modelname = models_folder + '/' + file

    num_rank_results = []
    
    # Test the model on different devices
    for folder_num in range(6, 11):
        data_path = f'Data/Test/D{folder_num}'

        Traces, key_interest, cts_interest = load_data(num_avg, data_path, interest_byte, shuffle=True)
        num_rank = test(modelname, Traces, key_interest, cts_interest, num_trace)
        
        print(f"Result for D{folder_num}: {num_rank}")
        num_rank_results.append(num_rank)
    
    # Get the average PGE result
    average_num_rank = sum(num_rank_results) / len(num_rank_results)
    print(f"\nAverage num_rank across D6-D10: {average_num_rank}")
