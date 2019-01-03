from __future__ import division
from __future__ import print_function
import sys
import os

#intended to be used for stripping sequences to equal lengths so they can be displayed as a sequence logo
def main():
    if len(sys.argv) < 2:
        print("Please specify a file")
        return
    for directory in sys.argv[1:]:
        for file in os.listdir(directory):
            lines = open(directory + file).readlines()
            seqs = []
            sequence = ""
            lexiconseq = []
            name = ""
            num_selenocysteine = 0
            num_internal_stop = 0
            num_total = 0
            valid = False
            
            for j in range(0, len(lines), 3):
                name = lines[j].split(' ')[0][1:]
                sequence = lines[j+1].strip()
                comment = lines[j+2][1:].strip()
                idx = comment.find('i')
                if idx > 5 and idx < len(sequence) - 20:
                    print(">" + name)
                    print(sequence[idx-5:idx+20])

    
if __name__ == "__main__":
    main()