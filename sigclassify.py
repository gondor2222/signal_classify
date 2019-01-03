from __future__ import division
from __future__ import print_function
import sys
import os
import numpy as np
from hmmlearn import hmm


lexicon = { 
'R': 0, 'H': 0, 'K': 0, # positive
    'D': 1, 'E': 1, #negative
    'S': 2, 'T': 2, 'N': 3, 'Q': 3, #polar uncharged
    'G': 4, 'P': 5, #special
    'C': 4, 'A': 6, 'V': 6, 'I': 7, #hydrophobic
    'L': 9, 'M': 4, 'F': 9, 'Y': 8, 'W': 9, #hydrophobic
    'X': 10 #unknown
}
num_emissions = lexicon['X'] + 1
lim_seq = 150 #max characters to read from sequence

annot_to_state = {
    's': 0, #not present in original annotations    
    'S': 1, #not present in original annotations    
    'n': 2,
    'h': 3,
    'c': 4,
    'r': 5, #not present in original annotations    
    'v': 6, #not present in original annotations    
    'g': 7, #not present in original annotations    
    'C': 8,
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
    5: 'r',
    6: 'v',
    7: 'g',
    8: 'C',
    9: 'O',
    10: 'M',
}


d_neg_n_tm = "training_data/negative_examples/non_tm"
d_neg_tm = "training_data/negative_examples/tm"
d_pos_n_tm = "training_data/positive_examples/non_tm"
d_pos_tm = "training_data/positive_examples/tm"

example_directories = [d_neg_n_tm, d_neg_tm, d_pos_n_tm, d_pos_tm]

class Sequence:
    def __init__(self, name, seq, lexiconseq, label, stateseq):
        self.name = name
        self.seq = seq
        self.lexiconseq = lexiconseq
        self.label = label
        self.stateseq = stateseq
        
def printseqs(seq, stateseqstrs):
    chars_per_line = 80
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

def getseqs(file):
    lines = open(file).readlines()
    seqs = []
    sequence = ""
    lexiconseq = []
    name = ""
    num_selenocysteine = 0
    num_internal_stop = 0
    num_total = 0
    valid = False
    
    for line in lines:
        if line.strip() == 'Sequence unavailable':
            continue
        if line[0] == '>':
            if len(sequence) > 0 and valid:
                #print(name)
                #print(sequence)
                if len(lexiconseq) == 0:
                    print("WARNING: EMPTY SEQ AT:")
                    print(name)
                    print(sequence)
                    print(line)
                    return
                seqs.append(Sequence(name, sequence, lexiconseq, None, None))
                              
            name = line[1:].strip()
            sequence = ""
            lexiconseq = []  
            valid = True
            num_total += 1
        else:
            
            sline = line.strip()
            if len(sline) == 0:
                continue
            if sline[len(sline) - 1] == '*':
                sline = sline[0:len(sline) - 1]
            sequence = sequence + sline
            if sline.find('U') != -1:
                num_selenocysteine += 1
                valid = False
                continue
            elif sline.find('*') != -1:
                if (valid):
                    num_internal_stop += 1
                valid = False
                continue
            for character in sline:
                em = lexicon[character]
                lexiconseq.append(em)
    print("{0} seqs with U, {1} with internal stop, of {2} valid, of {3} sequences".format(num_selenocysteine, num_internal_stop, len(seqs), num_total))
    return seqs
        
def get_model(directories, pos, tm):
    validation_fraction = 0.2
    start = np.zeros(num_states);
    transitions = np.zeros([num_states, num_states])
    emissions = np.zeros([num_states, num_emissions])
    model = hmm.MultinomialHMM(n_components=num_states)
    examples = []
    validations = []
    for i in range(len(directories)):
        #print(directories[i])
        for file in os.listdir(directories[i]):
            lines = open(directories[i] + "/" + file, 'r').readlines()
            for j in range(0, len(lines), 3):
                if len(lines[j].strip()) == 0 or lines[j].strip()[0] != '>':
                    print("Error: Line is invalid FASTA accession line:")
                    print(lines[j].strip())
                    print("In file \"" + directories[i] + "/" + file + "\"")
                    return False, None, None, None
                name = lines[j].strip().split(' ')[0][1:]
                sequence = lines[j+1].strip()
                comment = lines[j+2][1:].strip()   
                if len(sequence) == 0:
                    print("Error: entry \"" + name + "\" has empty sequence")
                    print("In file \"" + directories[i] + "/" + file + "\"")
                    return False, None, None, None
                if len(comment) != len(sequence):
                    print("Error: entry \"" + name + "\" has comment and sequence of different lengths")
                    print("In file \"" + directories[i] + "/" + file + "\"")
                    return False, None, None, None
                sequence = sequence[0:lim_seq]
                comment = comment[0:lim_seq] 
                stateseq = []
                lexiconseq = []
                
                prevstate = None
                is_validation = np.random.rand() < validation_fraction
                for k in range(0, len(sequence)):
                    if comment[k] not in annot_to_state:
                        print("Error: comment for entry \"" + name + "\" contains invalid state: " + comment[k])
                        print(comment)
                        print("In file \"" + directories[i] + "/" + file + "\"")
                        return False, None, None, None
                    state = annot_to_state[comment[k]]
                    letter = None
                    if k == 0:
                        letter = 's'
                    elif (k == 1 or k == 2) and pos:
                        letter = 'S'
                    elif k < len(comment) - 1 and (comment[k+1] == 'C'):
                        letter = 'g'
                    elif k < len(comment) - 2 and (comment[k+2] == 'C'):
                        letter = 'v'
                    elif (comment[k] == 'c' or comment[k] == 'n' or comment[k] == 'h') and (k == 8 or k == 7) and tm:
                        letter = 'r'
                    if letter != None:
                        state = annot_to_state[letter]
                        comment = comment[:k] + letter + comment[k+1:]
                    if sequence[k] not in lexicon:
                        print("Error: sequence for entry \"" + name + "\" contains invalid amino acid: " + sequence[k])
                        print(sequence)
                        print("In file \"" + directories[i] + "/" + file + "\"")
                        return False, None, None, None
                    em = lexicon[sequence[k]]
                    lexiconseq.append(em)
                    if not is_validation:
                        if k == 0:
                            start[state] += 1
                        else:
                            transitions[prevstate][state] += 1
                        emissions[state][em] += 1                        
                    prevstate = state
                    stateseq.append(state)
                seq = Sequence(name, sequence, lexiconseq, comment, stateseq)
                
                
                if is_validation:
                    validations.append(seq)
                else:
                    examples.append(seq)
    
    sum_trans = np.sum(transitions, 1)[:, None]
    sum_ems = np.sum(emissions, 1)[:, None]
    for i in range(num_states): #change all-0 rows to all-1 rows
        if sum_trans[i] == 0:
            for j in range(num_states):
                transitions[i][j] = 1
        if sum_ems[i] == 0:
            for j in range(num_emissions):
                emissions[i][j] = 1
    sum_trans = np.sum(transitions, 1)[:, None]
    sum_ems = np.sum(emissions, 1)[:, None]
    
    start = start / np.sum(start)
    emissions = emissions / sum_ems
    transitions = transitions / sum_trans
    
    model.startprob_ = start
    model.emissionprob_ = emissions
    model.transmat_ = transitions

    #print(model.startprob_)
    #print(model.emissionprob_)    
    #print(model.transmat_)
    #print(model.n_components)
    
    return True, examples, validations, model

def predict(seq, models):
    obs = np.transpose([seq.lexiconseq])
    logprob1, seq_enc1 = models[0].decode(obs)
    logprob2, seq_enc2 = models[1].decode(obs)
    logprob3, seq_enc3 = models[2].decode(obs)
    logprob4, seq_enc4 = models[3].decode(obs)
    logprobs = [logprob1, logprob2, logprob3, logprob4]
    seqs = [seq_enc1, seq_enc2, seq_enc3, seq_enc4]
    state_seqs = [[],[],[],[]]
    stateseqstrs = ["","","",""]
    for j in range(4):
        for k in range(len(seq_enc1)):
            state_seqs[j].append(state_to_annot[seqs[j][k]])
        stateseqstrs[j] = ''.join(state_seqs[j])
        if logprobs[j] == float('-inf'):
            logprobs[j] == -10000
    label = np.argmax(logprobs)
    return logprobs[0] + logprobs[2] < logprobs[1] + logprobs[3]
    

    
def runvalidation(validation_sets, classlabels, models, num_neg, num_pos):
    debug = False
    if sum(len(v_set) for v_set in validation_sets) == 0:
        print("There are no validation samples.")
        return

    id_pos = 0 #number of true positives
    id_neg = 0 #number of true negatives
    for i in range(4):
        actual = classlabels[i]
        num_correct = 0
        num_incorrect = 0
        for seq in validation_sets[i]:
            obs = np.transpose([seq.lexiconseq])
            predicted = predict(seq, models)
            correct = (predicted == actual)
            if correct and actual:
                id_pos += 1
            elif correct and not actual:
                id_neg += 1
            num_incorrect += (1 - correct)
            num_correct += correct
            
            
            if not correct and debug:
                printseqs(seq, stateseqstrs)
        if num_correct + num_incorrect != 0:
            print("{0} correct, {1} incorrect. Accuracy {2}%".format(num_correct, num_incorrect, num_correct * 100 / (num_correct + num_incorrect)))
        else:
            print("Class has no samples")
    if num_neg != 0:
        print("Specificity {0}%".format(100 * id_neg / num_neg))
    if num_pos != 0:
        print("Sensitivity {0}%".format(100 * id_pos / num_pos))
        
def main():
    np.random.seed(1)
    
    if len(sys.argv) > 1 and sys.argv[1] == '-h':
        print("Usage: sigclassify.py [file to count signal peptides in]. Runs validation experiments if no file given.")
        return
    if not os.path.isdir(d_neg_n_tm):
        print(d_neg_non_tm + " does not exist.")
        return
    if not os.path.isdir(d_neg_tm):
        print(d_neg_tm + " does not exist.")
        return
    if not os.path.isdir(d_pos_n_tm):
        print(d_pos_non_tm + " does not exist.")
        return
    if not os.path.isdir(d_pos_tm):
        print(d_pos_tm + " does not exist.")
        return
    
    id_correct = 0
    id_wrong = 0
    
    success1, ex_neg_n_tm, val_neg_n_tm, model_neg_n_tm = get_model([d_neg_n_tm], False, False)
    success2, ex_pos_n_tm, val_pos_n_tm, model_pos_n_tm = get_model([d_pos_n_tm],  True, False)
    success3, ex_neg_tm,   val_neg_tm,   model_neg_tm =   get_model([d_neg_tm],   False,  True)
    success4, ex_pos_tm,   val_pos_tm,   model_pos_tm =   get_model([d_pos_tm],   False,  True)
    if not (success1 and success2 and success3 and success4):
        return
        
    models = [model_neg_n_tm, model_pos_n_tm, model_neg_tm, model_pos_tm]
    examples = [ex_neg_n_tm, ex_pos_n_tm, ex_neg_tm, ex_pos_tm]
    validation_sets = [val_neg_n_tm, val_pos_n_tm, val_neg_tm, val_pos_tm]
    
    num_neg = len(val_neg_n_tm) + len(val_neg_tm) #true negative count
    num_pos = len(val_pos_n_tm) + len(val_pos_tm) #true positive count

    if len(sys.argv) == 1:
        runvalidation(validation_sets, [False, True, False, True], models, num_neg, num_pos)
    if len(sys.argv) > 1:
        if not os.path.isfile(sys.argv[1]):
            print("Could not find file " + sys.argv[1])
            return
        seqs = []
        seqs = getseqs(sys.argv[1])
        numpredicted = 0
        tenths = len(seqs) // 10
        for i in range(len(seqs)):
            seq = seqs[i]
            if i % tenths == 0:
                print("Progress : {0}%".format(100 * i / len(seqs)))
            numpredicted += predict(seq, models)
        print("Predicted number of signal peptide proteins: " + str(numpredicted))

        
            
    
if __name__ == "__main__":
    main()