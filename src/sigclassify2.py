from __future__ import division
from __future__ import print_function
import sys
import os
import numpy as np
from hmmlearn import hmm


lexicon = { 
'R': 0, 'H': 0, 'K': 0, # positive
    'D': 1, 'E': 1, #negative
    'S': 9, 'T': 2, 'N': 3, 'Q': 3, #polar uncharged
    'G': 4, 'P': 8, #special
    'C': 5, 'A': 6, 'V': 7, 'I': 10, #hydrophobic
    'L': 11, 'M': 12, 'F': 12, 'Y': 10, 'W': 12, #hydrophobic
    'X': 12 #unknown
}
num_emissions = lexicon['X'] + 1
lim_seq = 150 #max characters to read from sequence

num_states = 20;

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

def get_model(directories):
    validation_fraction = 0.2
    emissions = []
    lengths = []
    model = hmm.MultinomialHMM(n_components=num_states)
    examples = []
    validations = []
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
                is_validation = np.random.rand() < validation_fraction
                if not is_validation:
                    lengths.append(len(sequence))
                
                for k in range(0, len(sequence)):
                    em = lexicon[sequence[k]]
                    lexiconseq.append(em)
                    if not is_validation:
                        emissions.append(em)
                seq = Sequence(name, sequence, lexiconseq, comment, comment)
                if is_validation:
                    validations.append(seq)
                else:
                    examples.append(seq)
    emissions = np.transpose([emissions])
    lengths = np.transpose(lengths)    

    model.fit(emissions, lengths)

    #print(model.startprob_)
    #print(model.emissionprob_)    
    #print(model.transmat_)
    #print(model.n_components)
    
    return examples, validations, model
        
        
def main():
    np.random.seed(1)
    debug = False
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

    ex_neg, val_neg, model_neg = get_model([d_neg_n_tm, d_neg_tm])
    ex_pos, val_pos, model_pos = get_model([d_pos_n_tm, d_pos_tm])
        
    examples = [ex_neg, ex_pos]
    validation_sets = [val_neg, val_pos]
    
    num_neg = len(val_neg) #true negative count
    num_pos = len(val_pos) #true positive count
    
    if num_neg + num_pos == 0:
        print("There are no validation samples.")
        return
    id_pos = 0 #number of true positives
    id_neg = 0 #number of true negatives
    negatives = []
    positives = []
    for i in range(2):
        actual = i % 2
        num_correct = 0
        num_incorrect = 0
        for seq in validation_sets[i]:
            obs = np.transpose([seq.lexiconseq])
            logprob1, seq_enc1 = model_neg.decode(obs)
            logprob2, seq_enc2 = model_pos.decode(obs)
            logprobs = [logprob1, logprob2]
            seqs = [seq_enc1, seq_enc2]

            label = np.argmax(logprobs)
            predicted = logprobs[0] < logprobs[1]
            correct = (predicted == actual)
            if correct and actual:
                id_pos += 1
            elif correct and not actual:
                id_neg += 1
            num_incorrect += (1 - correct)
            num_correct += correct
            
            if actual:
                positives.append([logprobs[0], logprobs[1]])
            else:
                negatives.append([logprobs[0], logprobs[1]])
            if not correct and debug:
                printseqs(seq, stateseqstrs)
        #print("{0} correct, {1} incorrect. Accuracy {2}%".format(num_correct, num_incorrect, num_correct * 100 / (num_correct + num_incorrect)))
    #print("Specificity {0}%. Sensitivity {1}%".format(100 * id_neg / num_neg, 100 * id_pos / num_pos))
    print_data = True
    if print_data:
        sys.stdout.write("positives = [")
        for i in range(len(positives)):
            for j in range(2):
                try:
                    sys.stdout.write(str(positives[i][j]))
                except TypeError as e:
                    print("#")
                    print(positives[i][j])
                    print("#")
                    return
                if j == 0:
                    sys.stdout.write(",")
            if i < len(positives) - 1:
                sys.stdout.write(";")
        sys.stdout.write("];")
        sys.stdout.write("negatives = [")
        for i in range(len(negatives)):
            for j in range(2):
                sys.stdout.write(str(negatives[i][j]))
                if j == 0:
                    sys.stdout.write(",")
            if i < len(negatives) - 1:
                sys.stdout.write(";")
        sys.stdout.write("];")
            
    
if __name__ == "__main__":
    main()