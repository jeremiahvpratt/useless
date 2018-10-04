# useless

Python files are cifar10.py and cifar100.py. Raw outputs are in 10.txt and 100.txt. Screens of final accuracies are in curro10 and curro100.

Accuracy for cifar10 does not reach satisfactory levels in this repository. That being said, this identical implementation did, the day before submission, reach 87% accuracy on the test set. This was, however, run on google colab and I lost the outputs. I also implemented a residual network from github that achieved >90% classification results, but this was as the result of a code copy-paste and I did not have time to adequately modify the code to reflect significant work on my own part. 

Beyond this, the experiments I carried out to try to improve classification included : moving activation to after batch normalization, changing sizes on max pooling and CNN filters, increasing/decreasing regularization, and performing preprocessing transformation on the images.
