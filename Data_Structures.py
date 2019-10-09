import numpy as np
import pandas as pd
from cytoolz import itertoolz
from pathlib import Path
import csv
import math

#loads cleaned childes data set such that it contains the following columns: Transcript_id,Speaker_code Age,
# Utterance(with punctuation). In addition it replaces punctuation names with the symbol.

def load_childesdb_data():

    #Creates a dictionary which contains an entry for each punctuation category and corresponding symbolic translation
    # found in CHILDES.

    utype2symbol = {'declarative': ' .',
                    'question': ' ?',
                    'imperative_emphatic': ' ?'}

    # file handle is an object for loading in data from file.
    f_handle = open("SelectP_Utterances.csv", 'r')
    # csv reader loads data in csv-appropriate format (it does all the parsing for you).
    row_dicts = csv.DictReader(f_handle)

    # collecting samples
    childesdb_data = []
    speaker_dict = {}
    for row_dict in row_dicts:

        if not row_dict['speaker_code'] in speaker_dict:
            speaker_dict[row_dict['speaker_code']] = 1  # if item is not found add an entry to the dictionary.
            speaker_dict[row_dict['speaker_code']] += 1  # If speaker code is already in the dictionary increase it by 1.

        try:
            boundary_symbol = utype2symbol[row_dict['type']]
            # print(boundary_symbol)
        except KeyError:
            boundary_symbol = ' .'  # if type !=  any of the items in the dictionary utype2symbol replace with ' .'
        utterance = row_dict['gloss'] + boundary_symbol  # adds punctuation to the 'gloss' utterance
        sample = [row_dict['transcript_id'], row_dict['speaker_code'], int(float(row_dict['target_child_age'])),
                utterance.split()]

        childesdb_data.append(sample)
    return childesdb_data  # list of samples e.g. [speaker, age, w1, w2, ... punctuation]

#-----------------------------------------------------------------------------------------------------------------------
def load_mcdi_data():

    mcdi_data = pd.read_csv("Age_count_proportion.csv")
    mcdi_data = pd.DataFrame(mcdi_data)
    target_words = pd.read_csv("Target_words.csv")

    return mcdi_data, target_words
#-----------------------------------------------------------------------------------------------------------------------
def create_age_data_structures(mcdi_data):
    #age_data = pd.read_csv('Age_count_proportion.csv')
    # create the empty data structures
    age_list = []
    age_index_dict = {}

    age_column = mcdi_data['age']
    age_column = pd.Series(age_column).unique()

    for row in age_column:
        age_list.append(row)

    counter = 0

    for i in range(len(age_list)):
        age_index_dict[age_list[i]] = counter
        counter += 1

    #print(age_list)
    #print(age_dict)
    return age_list, age_index_dict
# #-----------------------------------------------------------------------------------------------------------------------
#
def create_target_data_structures(target_words):
    target_word_list = []
    target_word_index_dict = {}

    #target_words= pd.read_csv('Target_words.csv')
    target_words_column = target_words['words']

    for row in target_words_column:
        target_word_list.append(row)
        target_word_list.sort()
    #unsure how to make list items lower_case
    #Now create a dictionary
    counter= 0

    for i in range(len(target_word_list)):
        target_word_list[i] = target_word_list[i].lower()
        target_word_index_dict[target_word_list[i].lower()]= counter

        counter +=1

    #print(target_word_list)
    #print(target_word_dict)
    return target_word_list, target_word_index_dict
# #-----------------------------------------------------------------------------------------------------------------------
def create_doc_data_structures(childesdb_data):

    #utterances = pd.read_csv('SelectP_Utterances.csv')
    #create a list
    doc_list = []
    doc_index_dict = {}
    doc_counter = 0

    for i in range(len(childesdb_data)):
        doc_id = childesdb_data[i][0]
        if doc_id not in doc_index_dict:
            doc_list.append(doc_id)
            doc_index_dict[doc_id] = doc_counter
            doc_counter += 1

    return doc_list, doc_index_dict
# #-----------------------------------------------------------------------------------------------------------------------
#
def create_word_data_structures(childesdb_data):
    # unsure of how to access utterance column in order to create the list and dictionary of unique words.
    word_list = []
    word_index_dict = {}
    word_counter = 0

    for i in range(len(childesdb_data)):
        utterance = childesdb_data[i][3]
        for token in utterance:
            token = token.lower()
            if token not in word_index_dict:
                word_list.append(token)
                word_index_dict[token] = word_counter
                word_counter += 1

    return word_list, word_index_dict

def target_frequency_age_matrix(target_index_dict, age_index_dict, childesdb_data):
    num_targets = len(target_index_dict)
    num_ages = len(age_index_dict)

    target_age_freq_matrix = np.zeros([num_targets, num_ages], int)
    cumulative_target_frequency_matrix = np.zeros([num_targets, num_ages], int)

    for i in range(len(childesdb_data)):
        utterance = childesdb_data[i][3]
        age = childesdb_data[i][2]

        for token in utterance:
            token = token.lower()
            # update the correct row and column of target_age_freq_matrix for the token and age
            if token in target_index_dict:
                target_age_freq_matrix[target_index_dict[token], age_index_dict[age]] += 1
            if token in age_index_dict:
                if age in age_index_dict:
                    target_age_freq_matrix[target_index_dict[token], age_index_dict[age]] +=1

    for i in range(num_targets):
        for j in range(num_ages):
            if j == 0:
                cumulative_target_frequency_matrix[i,j] = target_age_freq_matrix[i,j]
            else:
                cumulative_target_frequency_matrix[i,j] = target_age_freq_matrix[i,j] + cumulative_target_frequency_matrix[i,j-1]

    return target_age_freq_matrix, cumulative_target_frequency_matrix

def output_data(target_list, target_index_dict, age_list, age_index_dict, mcdi_data, target_age_freq_matrix, cumulative_target_frequency_matrix):
    MCDIp = []

    #MCDIp = mcdi_data.at[0,'per_produce']

    num_targets = len(target_list)
    num_ages = len(age_list)

    #give me the value in column mcdi where age=18 and target=mommy

    print(num_targets, num_ages, len(target_index_dict), len(age_index_dict))
    print(target_age_freq_matrix.shape)

    f = open('predict_mcdi.csv', 'w')
    f.write('age,target,MCDIp,freq,cumul_freq\n')
    for i in range(num_targets):
        for j in range(num_ages):
            age = age_list[j]
            target = target_list[i]
            freq = target_age_freq_matrix[i,j]
            cumulative_freq = cumulative_target_frequency_matrix[i,j]
            bool_ind = (mcdi_data['age'] == age) & (mcdi_data['definition'] == target)
            try:
                MCDIp = mcdi_data[bool_ind]['per_produce'].values[0]
            except IndexError:
                print('Did not find', target)
                print(age, target, freq, MCDIp)
            f.write('{},{},{},{},{}\n'.format(age, target, MCDIp, freq, cumulative_freq))
         #print(mcdi_data[i,j])
#------------------------------------------------------------------------------------------------------------------------

def co_occurence_matrix(target_index_dict, age_index_dict, childesdb_data):

    window_type = 'forward' # forward, backward, summed, concatenated
    window_size = 7
    window_weight = 'flat' # linear or flat
    PAD = '*PAD*'

    # The goal is to create a 3 dimensional array of the following x,y,z dimensions: MCDI words X MCDI words X Age
    num_targets = len(target_index_dict)
    num_ages = len(age_index_dict)
    cooc_matrix_by_age_list = []

    corpus_by_age_list = []
    for i in range(num_ages):
        corpus_by_age_list.append([])

    # Then specify what items (words) will be updating the correct row and columns.
    for i in range(len(childesdb_data)):
        utterance = childesdb_data[i][3]
        age = childesdb_data[i][2]
        age_index = age_index_dict[age]
        corpus_by_age_list[age_index] += utterance

    # now we are ready to start counting co-occurrences for each age
    for i in range(num_ages):
        cooc_matrix = np.zeros([num_targets, num_targets, num_ages], float)
        current_corpus = corpus_by_age_list[i]
        current_corpus =
        # for token in current_corpus:
        #     current_corpus.remove(".")
        #     assert '.' not in current_corpus

        if len(current_corpus) > 0:
            current_corpus += [PAD] * window_size  # add padding such that all co-occurrences in last window are captured
            windows = itertoolz.sliding_window(window_size, current_corpus)

            for w in windows:
                for word1, word2, dist in zip([w[0]] * (window_size - 1), w[1:], range(1, window_size)):
                    # increment
                    if word1 == PAD or word2 == PAD:
                        continue

                    if word1 not in target_index_dict:
                        continue

                    if word2 not in target_index_dict:
                        continue
                    word1_index = target_index_dict[word1]
                    word2_index = target_index_dict[word2]

                    if window_weight == "linear":
                        cooc_matrix[word1_index, word2_index] += window_size - dist
                    elif window_weight == "flat":
                        cooc_matrix[word1_index, word2_index] += 1
            # window_type
            if window_type == 'forward':
                final_matrix = cooc_matrix
            elif window_type == 'backward':
                final_matrix = cooc_matrix.transpose()
            elif window_type == 'summed':
                final_matrix = cooc_matrix + cooc_matrix.transpose()
            elif window_type == 'concatenate':
                final_matrix = np.concatenate((cooc_matrix, cooc_matrix.transpose()))
            else:
                raise AttributeError('Invalid arg to "window_type".')

            cooc_matrix_by_age_list.append(final_matrix)

    return cooc_matrix_by_age_list

# -----------------------------------------------------------------------------------------------------------------------
#def main():
childesdb_data = load_childesdb_data()
mcdi_data, target_words = load_mcdi_data()

age_list, age_index_dict= create_age_data_structures(mcdi_data)
target_list, target_index_dict = create_target_data_structures(target_words)
doc_list, doc_index_dict = create_doc_data_structures(childesdb_data)
word_list, word_index_dict = create_word_data_structures(childesdb_data)
target_age_freq_matrix, cumulative_target_frequency_matrix = target_frequency_age_matrix(target_index_dict, age_index_dict, childesdb_data)
cooc_matrix = co_occurence_matrix(target_index_dict, age_index_dict, childesdb_data)
#print(target_age_freq_matrix.shape)
#output_data(target_list, target_index_dict, age_list, age_index_dict, mcdi_data, target_age_freq_matrix, cumulative_target_frequency_matrix)
#-----------------------------------------------------------------------------------------------------------------------





