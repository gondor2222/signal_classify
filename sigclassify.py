from __future__ import division
import sys
import os
import numpy as np
from hmmlearn import hmm


lexicon = { 
'R': 0, 'H': 0, 'K': 0, # positive
    'D': 1, 'E': 1, #negative
    'S': 8, 'T': 2, 'N': 2, 'Q': 2, #polar uncharged
    'G': 4, 'P': 9, #special
    'C': 5, 'A': 5, 'V': 6, 'I': 7, #hydrophobic
    'L': 3, 'M': 3, 'F': 3, 'Y': 9, 'W': 8, #hydrophobic
    'X': 10 #unknown
} #0 positive, 1 negative, 2 polar uncharged, 3 special, 4 hydrophobic, 5 unknown. Cysteine counted as hydrophobic
num_emissions = lexicon['X'] + 1

annot_to_state = {
    'n': 0,
    'h': 1,
    'c': 2,
    'v': 3,
    'g': 4, #not present in original annotations    
    'C': 5,
    'f': 6, #not present in original annotations
    'O': 6,
    'o': 7,
    'i': 8,
    'M': 9,
}
num_states = annot_to_state['M'] + 1;

state_to_annot = {
    0: 'n',
    1: 'h',
    2: 'c',
    3: 'g',
    4: 'C',
    5: 'f',
    6: 'O',
    7: 'o',
    8: 'i',
    9: 'M',

}


d_neg_non_tm = "training_data/negative_examples/non_tm"
d_neg_tm = "training_data/negative_examples/tm"
d_pos_non_tm = "training_data/positive_examples/non_tm"
d_pos_tm = "training_data/positive_examples/tm"

example_directories = [d_neg_non_tm, d_neg_tm, d_pos_non_tm, d_pos_tm]
neg_non_tm = 0
neg_tm = 1
pos_non_tm = 2
pos_tm = 3

class Sequence:
    def __init__(self, name, seq, comment):
        self.name = name
        self.seq = seq
        self.comment = comment

def calc_probs(directories):
    pass

        
        
def main():
    np.random.seed(1)
    if not os.path.isdir(d_neg_non_tm):
        print(d_neg_non_tm + " does not exist.")
        return
    if not os.path.isdir(d_neg_tm):
        print(d_neg_tm + " does not exist.")
        return
    if not os.path.isdir(d_pos_non_tm):
        print(d_pos_non_tm + " does not exist.")
        return
    if not os.path.isdir(d_pos_tm):
        print(d_pos_tm + " does not exist.")
        return
    examples = [[], [], [], []]
    id_correct = 0
    id_wrong = 0
    
    start = np.zeros(num_states);
    model = hmm.MultinomialHMM(n_components=num_states)
    transitions = np.zeros([num_states, num_states])
    emissions = np.zeros([num_states, num_emissions])
    for i in range(len(example_directories)):
        print(example_directories[i])
        for file in os.listdir(example_directories[i]):
            lines = open(example_directories[i] + "/" + file, 'r').readlines()
            for j in range(0, len(lines), 3):
                seq = Sequence(lines[j].split(' ')[0][1:], lines[j+1].strip()[0:100], lines[j+2][1:].strip())
                examples[i].append(seq)
                prevstate = None
                for k in range(0, len(seq.seq)):
                    try:
                        state = annot_to_state[seq.comment[k]]
                        if seq.comment[k] == 'c' and (seq.comment[k+1] == 'C'):
                            state = annot_to_state['g']
                            mod_string = list(seq.comment)
                            mod_string[k] = 'g'
                            seq.comment = ''.join(mod_string)
                            #print(seq.name + ":" + seq.seq[k:k+3])
                        elif seq.comment[k] == 'c' and (seq.comment[k+2] == 'C'):
                            state = annot_to_state['v']
                            mod_string = list(seq.comment)
                            mod_string[k] = 'v'
                            seq.comment = ''.join(mod_string)
                            #print(seq.name + ":" + seq.seq[k:k+3])
                        elif (seq.comment[k] == 'O' or seq.comment[k] == 'o') and (seq.comment[k-1] == 'C'):
                            state = annot_to_state['f']
                            mod_string = list(seq.comment)
                            mod_string[k] = 'f'
                            seq.comment = ''.join(mod_string)
                    except KeyError as e:
                        print(e)
                        print(seq.comment)
                        return
                    if k == 0:
                        start[state] += 1
                    else:
                        transitions[prevstate][state] += 1
                        emissions[state][lexicon[seq.seq[k]]] += 1
                        
                    prevstate = state
    #transitions[3] = np.ones([num_states])
    #print(transitions)
    sum_trans = np.sum(transitions, 1)[:, None]
    sum_ems = np.sum(emissions, 1)[:, None]
    
    sum_ems[sum_ems == 0] = 1
    transitions = transitions / sum_trans
    emissions = emissions / sum_ems
    start = start / np.sum(start)
    #print(start)
    #print(transitions)
    #print(emissions)
    model.startprob = start
    model.startprob_ = start
    model.emissionprob = emissions
    model.emissionprob_ = emissions
    model.transmat = transitions
    model.transmat_ = transitions
    #print(model.startprob)
    #print(model.n_components)
    #print(model.startprob_)
    
    for i in range(4):
        num_correct = 0
        num_incorrect = 0
        for seq in examples[i]:
            classes = []
            for character in seq.seq:
                classes.append(lexicon[character])
            
            logprob, state_seq_enc = model.decode(np.transpose([classes[0:100]]))
            state_seq = []
            for digit in state_seq_enc:
                state_seq.append(state_to_annot[digit])
            #print(seq.name)
            #print(''.join(state_seq))
            #print(seq.comment)
            predicted = state_seq[0] == 'n' and seq.seq[0] == 'M'
            correct = predicted == (i >= 2)
            num_incorrect += (1 - correct)
            num_correct += correct
            if not correct:
                #print(seq.name)
                #print(seq.seq)
                #print(''.join(state_seq))
                #print(seq.comment)
                pass
        print("{0} correct, {1} incorrect. Accuracy {2}%".format(num_correct, num_incorrect, num_correct * 100 / (num_correct + num_incorrect)))
        
            
    
if __name__ == "__main__":
    main()