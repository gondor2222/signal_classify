from __future__ import division
import sys
import os
import numpy as np
from hmmlearn import hmm


lexicon = { 
'R': 0, 'H': 0, 'K': 0, # positive
    'D': 1, 'E': 1, #negative
    'S': 8, 'T': 2, 'N': 2, 'Q': 2, #polar uncharged
    'G': 4, 'P': 10, #special
    'C': 5, 'A': 5, 'V': 6, 'I': 7, #hydrophobic
    'L': 3, 'M': 3, 'F': 3, 'Y': 9, 'W': 8, #hydrophobic
    'X': 12 #unknown
} #0 positive, 1 negative, 2 polar uncharged, 3 special, 4 hydrophobic, 5 unknown. Cysteine counted as hydrophobic
num_emissions = lexicon['X'] + 1

annot_to_state = {
    's': 0,
    't': 1,
    'S': 2,
    'n': 3,
    'h': 4,
    'c': 5,
    'V': 6,
    'v': 7,
    'g': 8, #not present in original annotations    
    'C': 9,
    'f': 10, #not present in original annotations
    'O': 11,
    'o': 12,
    'i': 13,
    'M': 14,
}
num_states = annot_to_state['M'] + 1;

state_to_annot = {
    0: 's',
    1: 't',
    2: 'S',
    3: 'n',
    4: 'h',
    5: 'c',
    6: 'V',
    7: 'v',
    8: 'g',
    9: 'C',
    10: 'f',
    11: 'O',
    12: 'o',
    13: 'i',
    14: 'M'

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
    def __init__(self, name, seq, lexiconseq, label, stateseq):
        self.name = name
        self.seq = seq
        self.lexiconseq = lexiconseq
        self.label = label
        self.stateseq = stateseq

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
    lim_seq = 400 #max characters to read from sequence
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
                name = lines[j].split(' ')[0][1:]
                sequence = lines[j+1].strip()[0:lim_seq]
                stateseq = []
                lexiconseq = []
                comment = lines[j+2][1:].strip()                
                prevstate = None
                
                for k in range(0, len(sequence)):
                    try:
                        state = annot_to_state[comment[k]]
                        if k == 0 and i >= 2:
                            letter = 's'
                            state = annot_to_state[letter]
                            comment = comment[:k] + letter + comment[k+1:]
                        elif k == 1 and i >= 2:
                            letter = 'S'
                            state = annot_to_state[letter]
                            comment = comment[:k] + letter + comment[k+1:]
                        elif (comment[k] == 'i') and len(comment) > k+1 and comment[k+1] != comment[k]:
                            letter = 't'
                            state = annot_to_state[letter]
                            comment = comment[:k] + letter + comment[k+1:]
                        elif comment[k] == 'c' and (comment[k+1] == 'C'):
                            state = annot_to_state['g']
                            comment = comment[:k] + 'g' + comment[k+1:]
                        elif comment[k] == 'c' and (comment[k+2] == 'C'):
                            state = annot_to_state['v']
                            comment = comment[:k] + 'v' + comment[k+1:]
                        elif (comment[k] == 'c' or comment[k] == 'h' or comment[k] == 'n') and (k == 10):
                            state = annot_to_state['V']
                            comment = comment[:k] + 'V' + comment[k+1:]
                        elif (comment[k] == 'O' or comment[k] == 'o') and (comment[k-2] == 'C'):
                            state = annot_to_state['f']
                            comment = comment[:k] + 'f' + comment[k+1:]
                    except KeyError as e:
                        print(e)
                        print(comment)
                        return
                    if k == 0:
                        start[state] += 1
                    else:
                        transitions[prevstate][state] += 1
                    em = lexicon[sequence[k]]
                    lexiconseq.append(em)
                    emissions[state][em] += 1                        
                    prevstate = state
                    stateseq.append(state)
                seq = Sequence(name, sequence, lexiconseq, comment, stateseq)
                examples[i].append(seq)
    #transitions[3] = np.ones([num_states])
    #print(transitions)
    sum_trans = np.sum(transitions, 1)[:, None]
    sum_ems = np.sum(emissions, 1)[:, None]
    
    sum_ems[sum_ems == 0] = 1
    transitions = transitions / sum_trans
    emissions = emissions / sum_ems
    start = start / np.sum(start)
    #print(transitions)
    #print(emissions)
    model.startprob = start
    model.startprob_ = start
    
    print(model.startprob)
    print(model.startprob_)
    model.emissionprob = emissions
    model.emissionprob_ = emissions
    model.transmat = transitions
    model.transmat_ = transitions
    #print(model.startprob)
    #print(model.n_components)
    #print(model.startprob_)
    
    chars_per_line = 80
    for i in range(4):
        num_correct = 0
        num_incorrect = 0
        for seq in examples[i]:
            logprob, state_seq_enc = model.decode(np.transpose([seq.lexiconseq]))
            logprob2, state_seq_enc2 = model.decode(np.transpose([seq.lexiconseq[0:60]]))            
            state_seq = []
            for digit in state_seq_enc:
                state_seq.append(state_to_annot[digit])
            #print(seq.name)
            stateseq2 = state_to_annot[state_seq_enc2[0]]
            stateseqstr = ''.join(state_seq)
            #print(''.join(state_seq))
            #print(seq.label)
            predicted = state_seq[0] == 's' and stateseq2 == 's'
            correct = (predicted == (i >= 2))
            num_incorrect += (1 - correct)
            num_correct += correct
            if predicted:
                logprob3, state_seq_enc3 = model.decode(np.transpose([seq.lexiconseq[0:stateseqstr.find('C')]]))
                state_seq3 = []
                for digit in state_seq_enc3:
                    state_seq3.append(state_to_annot[digit])
                if not correct:
                    print(''.join(state_seq3))
            debug = False
            if not correct and debug:
                j = 0
                print(seq.name)                
                while (j < len(seq.seq)):
                    num_print = min(chars_per_line, len(seq.seq) - chars_per_line)
                    print(seq.seq[j:j+num_print])
                    print(stateseqstr[j:j+num_print])
                    print(seq.label[j:j+num_print])
                    print('')
                    j += chars_per_line
        print("{0} correct, {1} incorrect. Accuracy {2}%".format(num_correct, num_incorrect, num_correct * 100 / (num_correct + num_incorrect)))
        
            
    
if __name__ == "__main__":
    main()