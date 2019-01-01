from __future__ import division
import sys
import os
import numpy as np
from hmmlearn import hmm


lexicon = { 
'R': 0, 'H': 0, 'K': 0, # positive
    'D': 1, 'E': 1, #negative
    'S': 2, 'T': 2, 'N': 3, 'Q': 3, #polar uncharged
    'G': 4, 'P': 8, #special
    'C': 5, 'A': 5, 'V': 6, 'I': 7, #hydrophobic
    'L': 10, 'M': 10, 'F': 10, 'Y': 9, 'W': 10, #hydrophobic
    'X': 10 #unknown
} #0 positive, 1 negative, 2 polar uncharged, 3 special, 4 hydrophobic, 5 unknown. Cysteine counted as hydrophobic
num_emissions = lexicon['X'] + 1
lim_seq = 150 #max characters to read from sequence

annot_to_state = {
    's': 0,
    'S': 1,    
    'n': 2,
    'h': 3,
    'c': 4,
    'V': 5,
    'v': 6,
    'g': 7, #not present in original annotations    
    'C': 8,
    't': 8,
    'O': 9,
    'o': 9,
    'i': 9,
    'M': 10,
}
num_states = annot_to_state['M'] + 1;

state_to_annot = {
    0: 's',
    1: 'S',
    2: 'n',
    3: 'h',
    4: 'c',
    5: 'V',
    6: 'v',
    7: 'g',
    8: 'C',
    9: 'O',
    10: 'M',
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

def get_model(directories, pos, tm):
    start = np.zeros(num_states);
    transitions = np.zeros([num_states, num_states])
    emissions = np.zeros([num_states, num_emissions])
    model = hmm.MultinomialHMM(n_components=num_states)
    examples = []
    for i in range(len(directories)):
        print(directories[i])
        for file in os.listdir(directories[i]):
            lines = open(directories[i] + "/" + file, 'r').readlines()
            for j in range(0, len(lines), 3):
                name = lines[j].split(' ')[0][1:]
                sequence = lines[j+1].strip()[0:lim_seq]
                stateseq = []
                lexiconseq = []
                comment = lines[j+2][1:].strip()                
                prevstate = None
                
                for k in range(0, len(sequence)):
                    state = annot_to_state[comment[k]]
                    if k == 0:
                        letter = 's'
                        state = annot_to_state[letter]
                        comment = comment[:k] + letter + comment[k+1:]
                    elif (k == 1) and pos:
                        letter = 'S'
                        state = annot_to_state[letter]
                        comment = comment[:k] + letter + comment[k+1:]
                    elif (k == 2) and pos and not tm:
                        letter = 'S'
                        state = annot_to_state[letter]
                        comment = comment[:k] + letter + comment[k+1:]
                    elif comment[k] == 'c' and (comment[k+1] == 'C'):
                        state = annot_to_state['g']
                        comment = comment[:k] + 'g' + comment[k+1:]
                    elif comment[k] == 'c' and (comment[k+2] == 'C'):
                        state = annot_to_state['v']
                        comment = comment[:k] + 'v' + comment[k+1:]
                    elif (comment[k] == 'c' or comment[k] == 'n') and (k == 8 or k == 7) and tm:
                        state = annot_to_state['V']
                        comment = comment[:k] + 'V' + comment[k+1:]
                    elif (comment[k] == 'c') and tm:
                        letter = 't'
                        state = annot_to_state[letter]
                        comment = comment[:k] + letter + comment[k+1:]
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
                examples.append(seq)
    
    sum_trans = np.sum(transitions, 1)[:, None]
    for i in range(num_states):
        if sum_trans[i] == 0:
            for j in range(num_states):
                transitions[i][j] = 1
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
    
    #print(model.startprob)
    #print(model.startprob_)
    model.emissionprob = emissions
    model.emissionprob_ = emissions
    model.transmat = transitions
    model.transmat_ = transitions
    #print(model.startprob)
    #print(model.n_components)
    #print(model.startprob_)
    return examples, model
        
        
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
    
    id_correct = 0
    id_wrong = 0
    
    examples_neg_non_tm, model_neg_non_tm = get_model([d_neg_non_tm], False, False)
    examples_pos_non_tm, model_pos_non_tm = get_model([d_pos_non_tm], True, False)
    examples_neg_tm, model_neg_tm = get_model([d_neg_tm], False, True)
    examples_pos_tm, model_pos_tm = get_model([d_pos_tm], False, True)
    
    chars_per_line = 80
    num_correct = 0
    num_incorrect = 0
    examples = [examples_neg_non_tm, examples_pos_non_tm, examples_neg_tm, examples_pos_tm]
    
    num_neg = len(examples_neg_non_tm) + len(examples_neg_tm) #true negative count
    num_pos = len(examples_pos_non_tm) + len(examples_pos_tm) #true positive count
    id_pos = 0 #number of true positives
    id_neg = 0 #number of true negatives
    for i in range(4):
        for seq in examples[i]:
            obs = np.transpose([seq.lexiconseq])
            logprob,  state_seq_enc =  model_neg_non_tm.decode(obs)
            logprob2, state_seq_enc2 = model_pos_non_tm.decode(obs)
            logprob3, state_seq_enc3 = model_neg_tm.decode(obs)
            logprob4, state_seq_enc4 = model_pos_tm.decode(obs)
            logprobs = [logprob, logprob2, logprob3, logprob4]
            seqs = [state_seq_enc, state_seq_enc2, state_seq_enc3, state_seq_enc4]
            state_seqs = [[],[],[],[]]
            stateseqstrs = ["","","",""]
            for j in range(4):
                for k in range(len(state_seq_enc)):
                    state_seqs[j].append(state_to_annot[seqs[j][k]])
                stateseqstrs[j] = ''.join(state_seqs[j])
            label = np.argmax(logprobs)
            predicted = logprob + logprob3 < logprob2 + logprob4 or label % 2 == 1
            #predicted = label % 2 == 1
            actual = i % 2
            correct = (predicted == actual)
            if correct and actual:
                id_pos += 1
            elif correct and not actual:
                id_neg += 1
            num_incorrect += (1 - correct)
            num_correct += correct
            debug = False
            if not correct and debug:
                j = 0
                print(seq.name)                
                while (j < len(seq.seq)):
                    num_print = min(chars_per_line, len(seq.seq) - chars_per_line)
                    print(seq.seq[j:j+num_print])
                    print(stateseqstrs[0][j:j+num_print])
                    print(stateseqstrs[1][j:j+num_print])
                    print(stateseqstrs[2][j:j+num_print])
                    print(stateseqstrs[3][j:j+num_print])
                    print(seq.label[j:j+num_print])
                    print('')
                    j += chars_per_line
        print("{0} correct, {1} incorrect. Accuracy {2}%".format(num_correct, num_incorrect, num_correct * 100 / (num_correct + num_incorrect)))
    print("Specificity {0}%. Sensitivity {1}%".format(id_neg / num_neg, id_pos / num_pos))
        
            
    
if __name__ == "__main__":
    main()