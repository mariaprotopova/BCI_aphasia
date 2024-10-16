# BCI_aphasia

* data/ -- folder with the data (preprocessed data, epochs and graphs);
* How the model works:
1) takes the raw data
2) filters (0.1)
3) resamples (10 Hz)
4) concatenates channels (horizontally)
5) SW-LDA

  
    _see detailed description in the stats/example.png_
![example](https://github.com/mariaprotopova/BCI_aphasia/assets/102407628/0f6ebeda-4097-4878-99d1-5d51adec9d27)
  
* bci_classification_pilot.ipynb -- script with the initial trial of classification (old version);
* functions.py -- script with functions for the preprocessing of the eeg data and to plot graphs (old version);

