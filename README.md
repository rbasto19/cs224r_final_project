This repository contains all my code for the CS 224R final project. To reproduce the resuls, follow [Token-Mol]([Token-Mol](https://github.com/jkwang93/Token-Mol)) to get the data and install the necessary dependencies. To run DPO, do "python3 dpo_train.py", which will train the model with hyperparameters specified in the script (can also be passed as command line arguments) and save generated ligands for the Adenosine A2A receptor at each validation. 

Note: this repo has significant amount of base code from [Token-Mol](https://github.com/jkwang93/Token-Mol) and [AliDiff](https://github.com/MinkaiXu/AliDiff), but most of the DPO training code and all necessary modifcations to adapt it were mine (around 500 lines). 

