# BCI_aphasia

* data/ -- folder with the data (preprocessed data, epochs and graphs);
* ERPs/ -- folder that contains graphs of ERPs for each participants and averaged across all subjects -- performed with MNE functions;
* epr_std/ -- the same as ERPs/, but Mean +- standard_error of the mean are shown -- using custom functions;
* stats/ -- EPRs with the results of statistical testing (deviation from 0 AND difference between stim. and distr. conditions)
    _see detailed description in the stats/example.png_
* bci_classification_pilot.ipynb -- script with the initial trial of classification (old version);
* functions.py -- script with functions for the preprocessing of the eeg data and to plot graphs (old version);

Tasks:
1. To find documentation about the data preprocessing
2. To find documentation about the model training and data averaging
