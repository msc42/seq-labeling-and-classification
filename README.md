# seq-labeling-and-classification

## copyright
This project was created in my work time as employee of the Karlsruhe Institute of Technology (KIT).
This work has been supported by the German Federal Ministry of Education and Research (BMBF) under the project OML (01IS18040A).

## setup
```./1_setup.sh```

## training
sequence classification:
```./2_train_classification.sh```

sequence labeling:
```./4_train_seq_labeling.sh```


you can skip the training and download the models from <https://www.dropbox.com/sh/hovvnj3cky55psi/AAAZHk_OfXjuLAlHeFPoWOAua?dl=0>

## evaluation
you can change `MODEL_FOR_PREDICTION` to the model that you want to evaluate

evaluate classification:
```./3_evaluate_classification.sh```

evaluate sequence labeling for evaluation error correction detection:
```./5_evaluate_detection_with_seq_labeling.sh```

evaluate sequence labeling for evaluation error correction:
```./5_evaluate_seq_labeling.sh```
