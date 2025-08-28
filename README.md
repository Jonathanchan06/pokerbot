Thought process

27/8/2025 : Using random forest --> still working on classification, undersampling seems unneccessary considering lack of data, attempting to oversample using imblearn

28/8/2025 : Attempted undersampling and oversampling. Going to stop this project, simply not enough data for the rarer hands for undersampling and 
oversampling to be useful. f1 score did improve slightly from before (improving all hands) 

classification report:
              precision    recall  f1-score   support

           0     0.6550    0.7577    0.7026    501209
           1     0.5638    0.5581    0.5610    422498
           2     0.3760    0.0029    0.0058     47622
           3     0.4394    0.0165    0.0318     21121
           4     0.1396    0.0095    0.0178      3885
           5     0.8903    0.1423    0.2454      1996
           6     0.1094    0.0049    0.0094      1424
           7     0.0000    0.0000    0.0000       230
           8     0.0000    0.0000    0.0000        12
           9     0.0108    0.3333    0.0208         3

    accuracy                         0.6164   1000000
   macro avg     0.3184    0.1825    0.1595   1000000
weighted avg     0.5962    0.6164    0.5907   1000000
