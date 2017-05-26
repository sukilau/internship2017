improves https://github.com/ziff/internship2017/issues/2

# Submission to Challenge Task 1: Improve accuracy with GridSearch
 
 
## Challenge
* Demonstrate an understanding of hyperparameter optimization using sklearn GridSearch on a convolutional deep net against a simplified MNIST digit recognition by improving out-of-sample accuracy above 0.98398.
 
 
## Approach

* I first implement GridSearch using coarse grids on individual hyperparameters while holding all other parameters same as the base model. This gives me a rough idea which hyperparameters give more significant result to our problem and allow me to narrow down to a finer set of hyperparameters for GridSearch in next step. 

* Next, I conduct GridSearch on combinations of selected hyperparameters. It is observed that tweaking number of neurons, convolutional filter size, batch size and number of epochs could result a greater improvement on accuracy. The best parameter set gives test accuracy of 0.99757.  

* See *mnist-gridsearch.ipynb* for the implementation algorithm.
 
 
## Improvement on Accuracy
* The following combination of parameters gives the best result on test accuracy of 0.99757.
 
 | Hyperparameter                 | Value                                                | 
 | ------------------------------ |-----------------------------------------------------:|
 | no. of neurons                 | 32 (1st conv), 64 (2nd conv), 128 (fully connecetd)  | 
 | convolutional filter size      | 5x5                                                  |
 | batch size                     | 64                                                   |
 | no. of epochs                  | 20                                                   |
 | weight initialization          | unifrom                                              |
 | activation function            | ReLu                                                 |
 | optimizer                      | Adam                                                 |
 | dropout                        | 0.25 (fully-connected), 0.5 (output layer)           |


 
## Takeaway
* Tuning hyperparameters for deep neural nets is hardest among other ML algorithms as it is slow to train a deep neural net and there are numerours parameters. This makes GridSearch difficult for deep neural nets.
* To implement GridSearch efficiently, it is better to start with coarse grids of individual hyperparameters on a smaller dataset, followed by fine grids on combinations of selected hyperparameters. 

 
## Further Thoughts
* Implementation of GridSearch could go crazy for deep neural nets. Randomized search and Bayesian optimization could be more effective in hyperparameter tuning for convnet. See my blog post on [Tuning Hyperparameter for Deep Neural Nets](https://sukilau.github.io/posts/tunning-hyperparameter)
 
 
## Resources  
 * [How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras](http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)
 * [Stanford CS class CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-3)
 * [Practical recommendations for gradient-based training of deep architectures by Yoshua Bengio](https://arxiv.org/abs/1206.5533)
 * [Deep Learning by Ian Goodfellow, Yoshua Bengio and Aaron Courville](http://www.deeplearningbook.org/)
 
 
 
