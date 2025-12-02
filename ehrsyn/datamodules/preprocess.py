import h5py
import torch
import os
import tqdm
import pickle
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
from functools import partial

def floor_time(data, time_window=10, max_time_len=2):
    """
    Floors the input data to the nearest time window and optionally scales it down.
    
    Parameters:
    - data (array-like): The input time data to be floored.
    - time_window (int): The time window to floor the data to.
    - max_time_len (int): Determines the format of the returned data. If 3, data is split into
      hundreds, tens, and ones places. If 2 (default), data is split into tens and ones.

    Returns:
    - list: A list of floored (and optionally scaled) time data, split according to max_time_len.
    """
    
    # Floor the data to the nearest time window
    floored_data = np.floor(data / time_window).astype(int) * time_window
    # Scale down the data for further processing
    scaled_down_data = floored_data // 10
    
    # Split the scaled data based on max_time_len
    if max_time_len == 3:
        return np.array([[x // 100, (x % 100) // 10, x % 10] for x in scaled_down_data])
    else:
        return np.array([[x // 10, x % 10] for x in scaled_down_data])
    
def split_to_csv(h5_path):
    data = h5py.File(h5_path, "r")['ehr']
    seed2split ={
        0:[],
        1:[],
        2:[],
    }
    for key in tqdm.tqdm(sorted(list(data.keys()))):
        for seed in [0,1,2]:
            seed2split[seed].append(data[key].attrs[f"split_{seed}"])
    df = {'pid': data.keys(), 'seed0': seed2split[0], 'seed1': seed2split[1], 'seed2': seed2split[2]}
    csv_path = h5_path.replace(".h5", "_split.csv")
    pd.DataFrame.from_dict(df).to_csv(csv_path, index=False)
    return

def h52numpy(data_path, obs_size):
    data = h5py.File(data_path, "r")['ehr']

    if obs_size in [6, 12]:
        max_time_len = 2
    elif obs_size in [24, 36, 48]:
        max_time_len = 3
    else:
        raise ValueError(f"obs_size {obs_size} is not recognized.")
            
    # Using list comprehensions to make the loop more concise
    input_list, type_list, dpe_list, time_list, time_num_list = [], [], [], [], []
    
    for key in tqdm.tqdm(sorted(list(data.keys()))):
        input_list.append(data[key]['hi'][:, 0, :])
        type_list.append(data[key]['hi'][:, 1, :])
        dpe_list.append(data[key]['hi'][:, 2, :])

        time_list.append(data[key]['floored_time'][:])
        time_num_list.append(data[key]['time'][:])
        
    return np.stack(input_list), np.stack(type_list), np.stack(dpe_list), np.array(time_list, dtype=object), np.array(time_num_list, dtype=object)
    

def reduce_vocab(input_array, data_path):
    vocab = np.unique(input_array)
    word2id = {word: _id for _id, word in enumerate(vocab)}
    id2word = {value: key for key, value in word2id.items()}
    assert id2word[0] == 0

    word2id_path = data_path.replace(".h5", "_word2id.pkl")
    id2word_path = data_path.replace(".h5", "_id2word.pkl")

    with open(word2id_path, 'wb') as f:
        pickle.dump(word2id, f)
    with open(id2word_path, 'wb') as f:
        pickle.dump(id2word, f)
    print("Save map files at ", id2word_path, " Size: ", len(id2word))

    word2id = pd.read_pickle(word2id_path)
    id2word = pd.read_pickle(id2word_path)
    return word2id, id2word


def check(tokenizer, input_samples, type_samples, dpe_samples, id2word, idx=0):
    if len(input_samples.shape) == 2:
        for i,j,k in zip(input_samples[idx], type_samples[idx], dpe_samples[idx]):
            if i == 0:
                return      
            i = np.vectorize(id2word.get)(i)
            print(k,'\t',j,'\t',tokenizer.decode(i))            

    elif len(input_samples.shape) == 3:
        for i,j,k in zip(input_samples[idx], type_samples[idx], dpe_samples[idx]):
            i = np.vectorize(id2word.get)(i)
            if (i[0] == 101):
                return
            for ii,jj,kk in zip(i, j, k):
                print(kk,'\t',jj,'\t',tokenizer.decode(ii))  
                if ii == 0:
                    return
    else:
        raise AssertionError("Wrong input shape")
    return

def process_slice(slice_data, word2id):
    return np.vectorize(word2id.get)(slice_data)

def main(data_path, dataset, obs_size):
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    if os.path.exists(data_path.replace(".h5", "_hi_input.npy")):
        inputs = np.load(data_path.replace(".h5", "_hi_input.npy"))
        types = np.load(data_path.replace(".h5", "_hi_type.npy"))
        dpes = np.load(data_path.replace(".h5", "_hi_dpe.npy"))
        times = np.load(data_path.replace(".h5", "_hi_time.npy"))
        num_times = np.load(data_path.replace(".h5", "_hi_num_time.npy"))
    else:
        split_to_csv(data_path)
        
        inputs, types, dpes, times, num_times = h52numpy(data_path, obs_size)
        np.save(data_path.replace(".h5", "_hi_input.npy"), inputs.astype(np.int16))
        np.save(data_path.replace(".h5", "_hi_type.npy"), types.astype(np.int16))
        np.save(data_path.replace(".h5", "_hi_dpe.npy"), dpes.astype(np.int16))
        np.save(data_path.replace(".h5", "_hi_time.npy"), times)
        np.save(data_path.replace(".h5", "_hi_num_time.npy"), num_times)
        print(inputs.shape, types.shape, dpes.shape, times.shape)
    
    word2id_path = data_path.replace(".h5", "_word2id.pkl")    
    id2word_path = data_path.replace(".h5", "_id2word.pkl")   
    if os.path.exists(word2id_path):
        word2id = pd.read_pickle(word2id_path)
        id2word = pd.read_pickle(id2word_path)
    else:
        word2id, id2word = reduce_vocab(inputs, data_path)
    print(dataset, obs_size, len(id2word))


    if os.path.exists(data_path.replace(".h5", "_hi_input_reduced.npy")):
        inputs_reduced = np.load(data_path.replace(".h5", "_hi_input_reduced.npy"))
    else:
        num_cores = cpu_count()
        slices = np.array_split(inputs, num_cores)
        process_slice_partial = partial(process_slice, word2id=word2id)

        # Process slices in parallel
        with Pool(processes=num_cores) as pool:
            results = list(tqdm.tqdm(pool.imap(process_slice_partial, slices), total=len(slices)))
        
        # Combine the results
        inputs_reduced = np.concatenate(results, axis=0)
        print("vectorizing is done... ")    

        np.save(data_path.replace(".h5", "_hi_input_reduced.npy"), inputs_reduced.astype(np.int16))
    
    check(tokenizer, inputs_reduced, types, dpes, id2word)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--obs_size", type=int, required=True)
    args = parser.parse_args()
    main(args.data_path, args.dataset, args.obs_size)