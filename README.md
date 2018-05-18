# Tensorflow Eager Execution

> Tensorflow Eager Execution mode allows an imperative programming style, similar to Numpy in addition to nearly all of the Tensorflow graph APIs, higher level APIs to build models (Keras) as well as easy debugging with the Python debug bridge.

Since Eager Execution APIs are quite recent, some kinks still exist, but as of this moment, they are minor and can be sidesteped. **These issues are highlighted in the notebooks and it is advised to browse through the comments, even if the topic is easy, so as to understand the limitations of Eager as TF 1.8**.

The following set of examples show usage of higher level APIs of Keras, different ways of performing the same thing, some issues that can arise and how to sidestep them while we wait for updates in Tensorflow to fix them.

It is to be noted, that I try to replicate most parts of this excellent [PyTorch Tutorial Set](https://github.com/yunjey/pytorch-tutorial). A few topics are missing - such as GANs and Image Captioning since I do not have the computational resources to train such models. A notable exception is Style Transfer, for which I have another repository dedicated to it, so I won't be porting it to Eager.

A final note : 
- Eager is evolving rapidly, and almost all of these issues that I stated here are edge cases that can/will be resolved in a later update. I still appreciate Eager, even with its limitations, as it offers a rich set of APIs from its Tensorflow heritage in an imperative execution environment like PyTorch. 
- This means that once the Eager API has all of its kinks ironed out, it will result in cleaner, more concise code and hopefully at performance close to Tensorflow itself.

# Table of Contents

Provided are two links - The title link leads to the local Jupyter Notebook where as the Colab Link Leads to a Colaboratory notebook. Please ensure to copy the contents of the notebook and run the command `!pip install --upgrade tensorflow` at the top before any of the imports to ensure you are using the latest Tensorflow version

**1. Basics**
  - [Linear Regression](https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/01_linear_regression.ipynb) - [Colab](https://colab.research.google.com/github/titu1994/tf-eager-examples/blob/master/notebooks/01_linear_regression.ipynb#scrollTo=ds1j7EC2ugi_)
  - [Logistic Regression](https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/02_logistic_regression.ipynb) - [Colab](https://colab.research.google.com/github/titu1994/tf-eager-examples/blob/master/notebooks/02_logistic_regression.ipynb)
  - [Feed Forward Neural Network](https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/03_feedforward_network.ipynb) - [Colab](https://colab.research.google.com/github/titu1994/tf-eager-examples/blob/master/notebooks/03_feedforward_network.ipynb)
  
**2. Intermediate**
  - [CNN (Simple)](https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/04_01_cnn.ipynb) - [Colab](https://colab.research.google.com/github/titu1994/tf-eager-examples/blob/master/notebooks/04_01_cnn.ipynb)
  - [CNN (Blocks)](https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/04_02_cnn_block.ipynb) - [Colab](https://colab.research.google.com/github/titu1994/tf-eager-examples/blob/master/notebooks/04_02_cnn_block.ipynb)
  - [Mini Inception](https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/05_inception.ipynb) - [Colab](https://colab.research.google.com/github/titu1994/tf-eager-examples/blob/master/notebooks/05_inception.ipynb)
  - [Mini ResNet](https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/05_resnet.ipynb) - [Colab](https://colab.research.google.com/github/titu1994/tf-eager-examples/blob/master/notebooks/05_resnet.ipynb)
  - [RNN (Canonical)](https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/06_01_rnn.ipynb) - [Colab](https://colab.research.google.com/github/titu1994/tf-eager-examples/blob/master/notebooks/06_01_rnn.ipynb)
  - [RNN (Custom)](https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/06_02_custom_rnn.ipynb) - [Colab (CPU)](https://colab.research.google.com/drive/17mOUhdWhbFAFwRxKiWm5GF5c8_PedsXX)
  - [RNN (Fast Hybrid)](https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/06_03_fast_rnn.ipynb) - [Colab](https://colab.research.google.com/github/titu1994/tf-eager-examples/blob/master/notebooks/06_03_fast_rnn.ipynb)
  - [Bi-Directional RNN (Canonical)](https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/07_01_bidirectional_rnn.ipynb) - [Colab](https://colab.research.google.com/github/titu1994/tf-eager-examples/blob/master/notebooks/07_01_bidirectional_rnn.ipynb)
  - [Bi-Directional RNN (Custom)](https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/07_02_custom_bidirectional_rnn.ipynb) - [Colab](https://colab.research.google.com/drive/1GZ1W35o8XKiQVFH5O2hkNmkfAaogHpay)
  - [RNN Language Model (Canonical)](https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/08_01_rnn_lm.ipynb) - [Colab](https://colab.research.google.com/drive/1uwCsSrg5PLXo6KZgHvZWTK92JGshV0Mq)
  - [RNN Language Model (Custom)](https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/08_02_rnn_lm.ipynb) - [Colab](https://colab.research.google.com/drive/1BiAlvJzZF5whWyLiFAPcj_QO24ScKERW)
  
**3. Advanced**
  - [Variational Autoencoders](https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/09_vae.ipynb) - [Colab](https://colab.research.google.com/github/titu1994/tf-eager-examples/blob/master/notebooks/09_vae.ipynb)
  - [**Models with Custom Variables**](https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/10_custom_models.ipynb) - [Colab](https://colab.research.google.com/github/titu1994/tf-eager-examples/blob/master/notebooks/10_custom_models.ipynb)
  
  
  
  
