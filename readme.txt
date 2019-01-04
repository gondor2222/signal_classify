to run the classifier on the example data:
$ python src/sigclassify.py <num_seeds>
By default runs with 20% validation samples. Changing this proportion currently requires changing the validation_fraction parameter in get_model
The tests will run num_seed times, and the 95% confidence intervals for the specificity and sensitivity will be printed. More seeds means smaller confidence intervals.


If any additional arguments are provided, they are expected to be files to count predicted signal peptides in, e.g.:
$ python src/sigclassify.py 1 proteomes/drosophila_all.faa
    will print how many sequences there are in this file, how many are valid fasta sequences, and how many of those are predicted to contain signal peptides.
A sequence is considered in valid if it contains anything other than the 21 "classic amino acids" or X. That is, sequences with selenocysteine or pyrrolysine will be rejected.

Note that you will have to download the proteomes yourself; they are too big to be provided in this project. See the readme in the proteomes folder for instructions.
