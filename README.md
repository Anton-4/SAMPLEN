## thesis abstract

Tree based ensemble methods like Random Forest and Extreme Gradient Boosting
are widely used because of their ease-of-use, fast training and excellent
performance. Keeping these properties in mind we set out to develop a better method.
We wanted to combine the strength of ensembling with diverse neural networks to challenge
tree based ensembles on datasets where they outperformed neural networks.
Through extensive experimentation and analysis we created SAMPLEN: "SAMPLing Ensemble Neural nets".
We use bagging and boosting with neural nets as the base learners, depending on the dataset the
neural nets are either highly overfitted or trained with early stopping.
Through well configured random search we are able to efficiently
optimize the architectures and hyperparameters for this ensemble of neural networks.
SAMPLEN outperforms Random Forest on three out of four tested datasets without sacrificing practical usability.

## Acknowledgement

SAMPLEN is based on an algorithm developed by Joeri Ruyssinck.
