# GReinSS

## Overview

GReinSS is a technique for learning discrete latent states from indirect observation data. 
This code provides a general implementation of GReinSS that is effective for all latent states that can in some form be represented as binary vectors. 

## Requirements

The required packages are numpy, scipy, and pytorch. 

## Usage 

First, load in the observations. We utilize the very small example from our set simulation with universe size 10. 

```python
from sharedGen import loadnpz
observations = loadnpz('./setExampleObservations.npz') #observations stored as a numpy array
```

Next, define the probability distribution Pr(X|S) of observations given states. This is an example from our set simulation with Gaussian noise and standard deviation 0.3. 
```python
import numpy as np
def log_calculate_pr_x_given_g(state, observation):
    #Inputs a state S and observation X and outputs log(Pr(X | S)). 
    return -1 * (np.sum(  0.5 * (observation - state) ** 2  ) / (0.3 ** 2))
```

Next, define the generative model. We utilize a model with the same architecture as in the manuscript. 
```python
from sharedGen import GeneratorNet
stateSize = 10 #size of the latent states
HiddenNeurons = 50 #the number of hidden neurons in the neural network
model = GeneratorNet(stateSize, HiddenNeurons)
```

Then, put all of these objects and some hyperparameters into a class to pass to the training function. 
```python
from sharedGen import gClass
ruleObject = gClass() #initialize problem class
ruleObject.log_calculate_pr_x_given_g = log_calculate_pr_x_given_g #enter Pr(X|S)
ruleObject.graphSize = stateSize #Set the size of states generated
ruleObject.observations_batch = observations #Provide the observation data
ruleObject.batchSize = 1000 #Batch size (GReinSS works best with large batch sizes)
ruleObject.learning_rate = 1e-3 #Learning rate
ruleObject.model = model #The generative model
```

Now one can train the model on this problem. 
```python
from sharedGen import simpleTrainModel
model_filename = './example_model.pt' #The file to store the pytorch model
simpleTrainModel(ruleObject, model_filename=model_filename)
```

After training, one can perform inference. 
```python
from sharedGen import simpleInference
predictedStates = simpleInference(ruleObject, model_filename)
```

### Graph inference example

As another example beyond our set inference simulations, one could utilize our graph inference simulations. 
First, we load in observations for the graph inference problem with 1000 graphs each with 100 random walks. 
```python
from sharedGen import loadnpz
observations = loadnpz('./graphExampleObservations.npz')
```

Then, load in the probability function Pr(X|S).

```python
from sharedGen import sim1_log_calculate_pr_x_given_g 
log_calculate_pr_x_given_g = sim1_log_calculate_pr_x_given_g
```

Next, define the generative model. We utilize a model with the same architecture as in the manuscript. 
```python
from sharedGen import GeneratorNet
stateSize = 90 #size of the latent states
HiddenNeurons = 50 #the number of hidden neurons in the neural network
model = GeneratorNet(stateSize, HiddenNeurons)
```

Then, put all of these objects and some hyperparameters into a class to pass to the training function. 
```python
from sharedGen import gClass
ruleObject = gClass() #initialize problem class
ruleObject.log_calculate_pr_x_given_g = log_calculate_pr_x_given_g #enter Pr(X|S)
ruleObject.graphSize = stateSize #Set the size of states generated
ruleObject.observations_batch = observations #Provide the observation data
ruleObject.batchSize = 1000 #Batch size (GReinSS works best with large batch sizes)
ruleObject.learning_rate = 1e-3 #Learning rate
ruleObject.model = model #The generative model

#Optional computational speedup described in "Optional computational speedup"
from sharedGen import sim1_fast_multi
multi_x_given_g = sim1_fast_multi
ruleObject.multi_x_given_g = multi_x_given_g
```

Now one can train the model on this problem. 
```python
from sharedGen import simpleTrainModel
model_filename = './graph_model.pt' #The file to store the pytorch model
simpleTrainModel(ruleObject, model_filename=model_filename)
```

After training, one can perform inference. 
```python
from sharedGen import simpleInference
predictedStates = simpleInference(ruleObject, model_filename)
```






### Optional off-policy learning

Off-policy learning is an optional inclusion in the GReinSS software. A function for modifying the sampling procedure is formatted as below. 
```python
def offPolicyRule(stateList, state_index):   
    #State list contains the current states
    #state_index is a numpy array containing the index within the batch of each of the states passed to this function. 
    #For instance, if  state_index[0] = 2, then the first element in stateList is the second element in the batch of states being generated. 
    #setAllow: the bias towards adding each element in the set to 1. The final column setAllow[:, -1] is the bias towards terminating.
    #Values in setAllow can be set to negative infinity to disallow them.  
    setAllow = torch.zeros((stateList.shape[0], stateSize+1))
    return setAllow,  torch.zeros((graphList.shape[0], finalProbSize))
ruleObject.offPolicyRule = offPolicyRule
```

Then, training is modified to include off-policy sampling as below. 
```python
from sharedGen import simpleTrainModel
offPolicy = True
model_filename = './example_model.pt'
simpleTrainModel(ruleObject, model_filename=model_filename, offPolicy=offPolicy)
```

Similarly, inference is modified as below. 
```python
from sharedGen import simpleInference
offPolicy = True
model_filename = './example_model.pt'
predictedStates = simpleInference(ruleObject, model_filename, offPolicy=offPolicy)
```



### Optional computational speedup

Rather than defining a function for calculating Pr(X|S) for one observation and state at a time, it can be much more computationally efficient to use a vectorized function for computing Pr(X|S) given a list of states and a list of observations. As an example, for the set simulations one can use the following. 

```python
noise_level = 0.3
def multi_x_given_g(adjacency_matrices, obs_matrix):
    obs_matrix = torch.tensor(obs_matrix).float().to(adjacency_matrices.device)
    adjacency_matrices = adjacency_matrices.reshape((adjacency_matrices.shape[0], 1, adjacency_matrices.shape[1]))
    obs_matrix = obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1]))
    diff1 = 0.5 * torch.sum((adjacency_matrices - obs_matrix) ** 2 , axis=2)
    prob_mult = -1 * diff1 / (noise_level ** 2) 
    return prob_mult

ruleObject.multi_x_given_g = multi_x_given_g
```

For the graph simulations, one can use the following.

```python
from sharedGen import sim1_fast_multi
multi_x_given_g = sim1_fast_multi
ruleObject.multi_x_given_g = multi_x_given_g
```



