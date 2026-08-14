import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy
import time

import math
import os


import pandas as pd



import numpy as np
import pandas as pd




def loadnpz(name, allow_pickle=False):

    #This simple function more easily loads in compressed numpy files.

    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data


def uniqueValMaker(X):

    _, vals1 = np.unique(X[:, 0], return_inverse=True)

    for a in range(1, X.shape[1]):

        #vals2 = np.copy(X[:, a])
        #vals2_unique, vals2 = np.unique(vals2, return_inverse=True)
        vals2_unique, vals2 = np.unique(X[:, a], return_inverse=True)

        vals1 = (vals1 * vals2_unique.shape[0]) + vals2
        _, vals1 = np.unique(vals1, return_inverse=True)

    return vals1




import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------
# A custom linear layer that applies an elementwise mask
# ------------------------------------------------------
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask):
        """
        Args:
            in_features: size of each input sample.
            out_features: size of each output sample.
            mask: a binary tensor of shape (out_features, in_features)
                  that will be multiplied elementwise with the weight matrix.
        """
        super().__init__(in_features, out_features)
        self.register_buffer("mask", mask)

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


# ------------------------------------------------------
# Autoregressive Model using masked fully-connected layers
# ------------------------------------------------------
class AutoregressiveMatrixModel(nn.Module):
    def __init__(self, graphSize, numMean, Nhidden2):
        super().__init__()

        self.numDist = numMean
        self.graphSize = graphSize
        self.seq_length = graphSize + 1   # +1 for start token

        self.embed_dim = Nhidden2
        #self.embed_dim = 10
        self.start_token = numMean

        # hidden width
        Nhidden = self.seq_length * Nhidden2
        #Nhidden = self.seq_length * 10
        self.Nhidden = Nhidden

        #NHidden2 = 10
        #NHidden2 = 20
        NHidden2 = numMean
        self.Nhidden2 = NHidden2

        self.stdFactor = nn.Parameter(torch.zeros(  ( graphSize, numMean, 1 ) ))

        # numDist actual values + 1 start token
        self.embedding = nn.Embedding(numMean + 1, self.embed_dim)

        # --------------------------------------------------
        # Base autoregressive mask over token positions
        # token i can depend only on tokens < i
        # shape: (seq_length, seq_length)
        # --------------------------------------------------
        base_mask = torch.tril(
            torch.ones(self.seq_length, self.seq_length),
            diagonal=-1
        )

        # --------------------------------------------------
        # fc1: input is flattened embeddings
        # input shape = (batch, seq_length * embed_dim)
        # output shape = (batch, Nhidden)
        #
        # We assign hidden units to token positions by repeating rows.
        # --------------------------------------------------
        hidden_repeat = math.ceil(Nhidden / self.seq_length)

        # First expand rows so hidden units correspond to token positions
        mask_in_hidden_tokens = base_mask.repeat_interleave(hidden_repeat, dim=0)[:Nhidden]
        # Then expand columns so each token corresponds to embed_dim input columns
        mask_in_hidden = mask_in_hidden_tokens.repeat_interleave(self.embed_dim, dim=1)
        # shape: (Nhidden, seq_length * embed_dim)

        # --------------------------------------------------
        # fc2: output is seq_length * numDist logits
        # hidden units are grouped by token position, so expand columns
        # according to hidden_repeat
        # --------------------------------------------------
        #mask_hidden_out_base = base_mask.repeat_interleave(self.numDist, dim=0)
        mask_hidden_out_base = base_mask.repeat_interleave(NHidden2, dim=0)
        mask_hidden_out = mask_hidden_out_base.repeat_interleave(hidden_repeat, dim=1)[:, :Nhidden]
        # shape: (seq_length * numDist, Nhidden)

        input_dim = self.seq_length * self.embed_dim

        self.fc1 = MaskedLinear(input_dim, Nhidden, mask=mask_in_hidden)
        #self.fc2 = MaskedLinear(Nhidden, self.seq_length * numDist, mask=mask_hidden_out)
        self.fc2 = MaskedLinear(Nhidden, self.seq_length * NHidden2, mask=mask_hidden_out)

        self.fc3 = nn.Linear(NHidden2, numMean)

    def forward(self, x):
        """
        Args:
            x: LongTensor of shape (batch, graphSize) or (batch, seq_length)
        Returns:
            logits: FloatTensor of shape (batch, graphSize, numDist)
        """

        # If input does not yet include the start token, prepend it
        if x.shape[1] == self.graphSize:
            startPart = torch.full(
                (x.shape[0], 1),
                self.start_token,
                dtype=torch.long,
                device=x.device
            )
            x = torch.cat((startPart, x.long()), dim=1)
        else:
            x = x.long()

        batch_size = x.size(0)

        # Embed tokens: (batch, seq_length, embed_dim)
        x_emb = self.embedding(x)

        # Flatten token embeddings into one vector:
        # (batch, seq_length * embed_dim)
        x_flat = x_emb.reshape(batch_size, -1)

        # First masked layer
        h = self.fc1(x_flat)
        h = F.relu(h)

        # Output masked layer
        out = self.fc2(h)


        #out = out.view(batch_size * self.seq_length, self.Nhidden2)
        #out = self.fc3( F.relu(out) )
        


        # Reshape to (batch, seq_length, numDist)
        logits = out.view(batch_size, self.seq_length, self.numDist)

        # Remove start-token position from outputs
        logits = logits[:, 1:, :]

        logits = torch.log_softmax(logits, dim=2)



        if False:
            #Sets minimum probability 1/1000
            sum1 = torch.logsumexp(logits, axis=2)
            sum1 = sum1.reshape((sum1.shape[0], sum1.shape[1], 1))[:, :, np.zeros(logits.shape[2], dtype=int) ]
            #epsilon = sum1 - np.log(1000)
            epsilon = sum1 - np.log(100)
            #print (logits[0, 0])
            logits = torch.logaddexp(logits, epsilon)
            #print (logits[0, 0])
            #quit()

        
        return logits

    @torch.no_grad()
    def generate(self, batch_size=1):
        """
        Autoregressively generate a batch of new binary matrices.
        
        Args:
            device: The device on which to run generation.
            batch_size: Number of matrices to generate.
        
        Returns:
            generated: Tensor of shape (batch_size, matrix_size, matrix_size)
                    containing the generated binary matrices.
        """
        self.eval()
        # Start with a batch of start tokens (token index 2).
        generated = torch.full((batch_size, 1), 2, dtype=torch.long)#, device=device)
        
        # Generate until we have a complete sequence of tokens (seq_length tokens).
        for i in range(1, self.seq_length):
            # For positions not yet generated, pad with zeros.
            # (These padding tokens won't affect the output because of the mask.)

            
            if generated.size(1) < self.seq_length:
                #print ("HI1")
                pad = torch.zeros((batch_size, self.seq_length - generated.size(1)),
                                dtype=torch.long)#, device=device)
                cur = torch.cat([generated, pad], dim=1)
            else:
                #print ("HI2")
                cur = generated
            #print (cur.shape)
            logits = self.forward(cur)  # shape: (batch_size, seq_length, 2)
            # Use the logits at position i for each sequence in the batch.
            logits_i = logits[:, i, :]   # shape: (batch_size, 2)
            probs = F.softmax(logits_i, dim=-1)
            # Sample the next token for each sequence in the batch.
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (batch_size, 1)
            generated = torch.cat([generated, next_token], dim=1)


        #print ('generated')
        #print (generated.shape)
        # Remove the start token and reshape the tokens to matrices.
        # The generated tensor is of shape (batch_size, seq_length) with seq_length = matrix_size*matrix_size + 1.
        # We remove the first token and then reshape each sequence to (matrix_size, matrix_size).
        generated = generated[:, 1:]  # shape: (batch_size, matrix_size*matrix_size)
        #generated = generated.view(batch_size, self.matrix_size, self.matrix_size)



        

        
        return generated
    






class quick_autoregressive(nn.Module):
    def __init__(self, graphSize, numMean, Nhidden2):
        super().__init__()

        self.numDist = numMean
        self.graphSize = graphSize
        self.seq_length = graphSize + 1   # +1 for start token

        self.embed_dim = Nhidden2
        #self.embed_dim = 10
        self.start_token = numMean

        # hidden width
        Nhidden = self.seq_length * Nhidden2
        print ('Nhidden', Nhidden)
        #Nhidden = self.seq_length * 10
        self.Nhidden = Nhidden

        #NHidden2 = 10
        #NHidden2 = 20
        #NHidden2 = numMean
        self.Nhidden2 = Nhidden2

        self.stdFactor = nn.Parameter(torch.zeros(  ( graphSize, numMean, 1 ) ))

        # numDist actual values + 1 start token
        self.embedding = nn.Embedding(numMean + 1, self.embed_dim)

        # --------------------------------------------------
        # Base autoregressive mask over token positions
        # token i can depend only on tokens < i
        # shape: (seq_length, seq_length)
        # --------------------------------------------------
        base_mask = torch.tril(
            torch.ones(self.seq_length, self.seq_length),
            diagonal=-1
        )

        # --------------------------------------------------
        # fc1: input is flattened embeddings
        # input shape = (batch, seq_length * embed_dim)
        # output shape = (batch, Nhidden)
        #
        # We assign hidden units to token positions by repeating rows.
        # --------------------------------------------------
        hidden_repeat = math.ceil(Nhidden / self.seq_length)

        # First expand rows so hidden units correspond to token positions
        mask_in_hidden_tokens = base_mask.repeat_interleave(hidden_repeat, dim=0)[:Nhidden]
        # Then expand columns so each token corresponds to embed_dim input columns
        mask_in_hidden = mask_in_hidden_tokens.repeat_interleave(self.embed_dim, dim=1)
        # shape: (Nhidden, seq_length * embed_dim)

        # --------------------------------------------------
        # fc2: output is seq_length * numDist logits
        # hidden units are grouped by token position, so expand columns
        # according to hidden_repeat
        # --------------------------------------------------
        #mask_hidden_out_base = base_mask.repeat_interleave(self.numDist, dim=0)

        print ("LLL")
        print (Nhidden2)
        print (hidden_repeat)
        
        mask_hidden_out_base = base_mask.repeat_interleave(Nhidden2, dim=0)
        mask_hidden_out = mask_hidden_out_base.repeat_interleave(hidden_repeat, dim=1)[:, :Nhidden]
        # shape: (seq_length * numDist, Nhidden)

        input_dim = self.seq_length * self.embed_dim

        self.fc1 = MaskedLinear(input_dim, Nhidden, mask=mask_in_hidden)
        #self.fc2 = MaskedLinear(Nhidden, self.seq_length * numDist, mask=mask_hidden_out)
        self.fc2 = MaskedLinear(Nhidden, self.seq_length, mask=mask_hidden_out)


        self.finalLinear = nn.Parameter(torch.zeros(  ( 1, graphSize, numMean ) ))
        self.finalLinear_bias = nn.Parameter(torch.zeros(  ( 1, graphSize, numMean ) ))

        

    def forward(self, x):
        """
        Args:
            x: LongTensor of shape (batch, graphSize) or (batch, seq_length)
        Returns:
            logits: FloatTensor of shape (batch, graphSize, numDist)
        """

        # If input does not yet include the start token, prepend it
        if x.shape[1] == self.graphSize:
            startPart = torch.full(
                (x.shape[0], 1),
                self.start_token,
                dtype=torch.long,
                device=x.device
            )
            x = torch.cat((startPart, x.long()), dim=1)
        else:
            x = x.long()

        batch_size = x.size(0)

        # Embed tokens: (batch, seq_length, embed_dim)
        x_emb = self.embedding(x)

        # Flatten token embeddings into one vector:
        # (batch, seq_length * embed_dim)
        x_flat = x_emb.reshape(batch_size, -1)

        # First masked layer
        h = self.fc1(x_flat)
        h = F.relu(h)

        
        # Output masked layer
        out = self.fc2(h)[:, 1:]

        out = out.reshape(( out.shape[0], out.shape[1], 1 ))

        logits = (out * self.finalLinear) + self.finalLinear_bias



        #logits = out.view(batch_size, self.seq_length, self.numDist)

        # Remove start-token position from outputs
        #logits = logits[:, 1:, :]

        




        logits = torch.log_softmax(logits, dim=2)
        
        return logits

    


class quickPolicy(nn.Module):
    def __init__(self, graphSize, numMean, NHiddenReg):
        super().__init__()


        #self.Nhidden = 5
        #self.Nhidden = 10
        #self.Nhidden = 40
        self.Nhidden = NHiddenReg


        self.numDist = numMean
        self.graphSize = graphSize
        self.seq_length = graphSize + 1   # +1 for start token

        self.embedding = nn.Embedding(numMean, self.Nhidden)


        self.stdFactor = nn.Parameter(torch.zeros(  ( graphSize, numMean, 1 ) ))

        self.connection1 = nn.Parameter(torch.rand(  ( 1, graphSize, self.Nhidden, self.Nhidden ) ) * 0.001)
        self.connection2 = nn.Parameter(torch.rand(  ( 1, graphSize, self.Nhidden, self.numDist ) ) * 0.001)


        #self.connectionM = nn.Parameter(torch.rand(  ( 1, graphSize, self.Nhidden, self.Nhidden ) ) * 0.001)


        self.nonlin = F.relu


    def forward(self, x):

        x = self.embedding(x)

        x = x.reshape(( x.shape[0], x.shape[1], x.shape[2], 1 ))
        x = torch.sum(x * self.connection1, axis=2)
        x = torch.cumsum(x, axis=1)
        x = self.nonlin(x)


        if False:
            x = x.reshape(( x.shape[0], x.shape[1], x.shape[2], 1 ))
            x = torch.sum(x * self.connectionM, axis=2)
            x = torch.cumsum(x, axis=1)
            x = self.nonlin(x)




        x = x.reshape(( x.shape[0], x.shape[1], x.shape[2], 1 ))
        x = torch.sum(x * self.connection2, axis=2)


        


        logits = torch.log_softmax(x, dim=2)

        return logits



class Encoder(nn.Module):
    def __init__(self, graphSize, Nhidden, numDist):
        super(Encoder, self).__init__()
        self.input_size = graphSize
        self.hidden_size = Nhidden
        self.numDist = numDist
        self.output_size = graphSize * numDist
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)


        self.fc_m1 = nn.Linear(self.hidden_size, self.hidden_size)

        #Nhidden2 = 10
        #Nhidden2 = 30

        #self.w1 = nn.Parameter(torch.randn( 1, graphSize , self.numDist, Nhidden2  ))
        #self.b1 = nn.Parameter(torch.randn( 1, graphSize , Nhidden2  ))
        #self.w2 = nn.Parameter(torch.randn( 1, graphSize , Nhidden2, numDist  ))
        #self.b2 = nn.Parameter(torch.randn( 1, graphSize , numDist  ))

        #self.w1 = nn.Parameter(torch.randn( 1, 1 , self.numDist, Nhidden2  ))
        #self.b1 = nn.Parameter(torch.randn( 1, 1 , Nhidden2  ))

        #self.layernorm1 = nn.LayerNorm(self.hidden_size)



        self.nonlin = F.leaky_relu

    def forward(self, x):

        #print (x.shape)


        x1 = self.fc1(x)
        #print (x1.shape)
        #x1 = self.layernorm1(x1)

        #x1 = x1 * 0.1 #Added Now
        x1 = F.leaky_relu(x1)

        

        #x1 = F.leaky_relu(self.fc_m1(x1))


        logits = self.fc2(x1)
        logits = logits.reshape(( logits.shape[0],  self.input_size, self.numDist  ))

        #x_dist = x.reshape(( x.shape[0],  x.shape[1], 1, 1)) - (torch.arange(3 0) % 10).reshape((1, 1, self.numDist, 1))
        #x_dist = torch.abs(x_dist) * -1

        #x_dist[:, :, 10:] = -100


        #print (x[0, 0])
        #print (x_dist[0, 0])
        #quit()


        #logits = logits * 0

        #x2 = torch.sum( x.reshape(( x.shape[0],  x.shape[1], 1, 1)) + self.w1, axis=2) + self.b1
        #x2 = self.nonlin(x2)

        #x2 = torch.sum( x_dist + self.w1, axis=2) + self.b1
        #x2 = self.nonlin(x2)
        #x2 = torch.sum(x2.reshape((x2.shape[0], x2.shape[1], x2.shape[2], 1)) + self.w2, axis=2) + self.b2

        #logits = logits * 0.01 #Trying to put preference on x2

        #logits = logits * 0.001  #Used generally
        #logits = logits * 0.01 
        #logits = logits + x2
        #logits = x_dist
        #logits = x2

        if False:
            #Sets minimum probability 1/1000
            sum1 = torch.logsumexp(logits, axis=2)
            sum1 = sum1.reshape((sum1.shape[0], sum1.shape[1], 1))[:, :, np.zeros(logits.shape[2], dtype=int) ]
            epsilon = sum1 - np.log(1000)
            logits = torch.logaddexp(logits, epsilon)

        logits = torch.log_softmax(logits, axis=2)


        #cost = torch.exp(logits)
        #cost = cost * logits * -1
        #cost = torch.sum(cost, axis=2)
        #cost = cost.reshape((cost.shape[0], cost.shape[1] , 1))

        #logits = logits * 2 #Penalty for low probability states

        #logits = logits - (cost * 0.2)
        # 
        # logits = logits - (cost * 0.2)
        #         

        return logits
    



class SpecialEncoder(nn.Module):
    def __init__(self, graphSize, Nhidden, Ngroups, numDist):
        super(SpecialEncoder, self).__init__()
        self.input_size = graphSize
        self.hidden_size = Nhidden
        self.numDist = numDist
        self.Ngroups = Ngroups
        self.output_size = graphSize * numDist * Ngroups
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

        self.fc_group = nn.Linear(self.hidden_size, self.Ngroups)

        self.nonlin = F.leaky_relu

    def forward(self, x):

        x1 = self.fc1(x)
        x1 = F.leaky_relu(x1)
        logits = self.fc2(x1)

        groupProbs = self.fc_group(x1) * 0.01
        groupProbs = torch.log_softmax(groupProbs, axis=1)


        logits = logits.reshape(( logits.shape[0], self.Ngroups, self.input_size, self.numDist  ))  * 0.01
        logits = torch.log_softmax(logits, axis=3)

        return groupProbs, logits



def getProbSamples(probDist, samples):

    time2 = time.time()

    argAll = np.indices(samples.shape).reshape(len(samples.shape), -1).T
    #argAll = np.argwhere( np.zeros(samples.shape) > -1 )
    probDistPaste = torch.zeros(samples.shape)
    #print (time.time() - time2)
    #time2 = time.time()

    probDistPaste[argAll[:, 0], argAll[:, 1]] = probDist[argAll[:, 0], argAll[:, 1], samples[argAll[:, 0], argAll[:, 1]]]
    probDistPaste = torch.sum(probDistPaste, axis=1)

    #print (time.time() - time2)
    #quit()

    return probDistPaste


def getProbSamples_onehot(probDist, samples_oneHot, cumProb=False):

    

    selected = (probDist * samples_oneHot).sum(dim=2)  # (B, L)

    if cumProb:
        return selected
    else:

        # sum over sequence dimension
        return selected.sum(dim=1)  # (B,)



def getProbGroupSamples_onehot(probDist, groupProbs, samples_oneHot, cumProb=False):

    if cumProb:
        print ('cumprob not implemented!')
        quit()


    samples_oneHot = samples_oneHot.reshape(( samples_oneHot.shape[0], 1, samples_oneHot.shape[1], samples_oneHot.shape[2] ))
    selected = (probDist * samples_oneHot).sum(dim=(2, 3))  # (B, L)

    selected = selected + groupProbs

    selected = torch.logsumexp(selected, axis=1)

    return selected

def poisson_log_prob(k, rate, eps=1e-8):
    # k: observed counts (tensor)
    # rate: expected counts (same shape or broadcastable)
    rate = torch.clamp(rate, min=eps)  # avoid log(0)
    return k * torch.log(rate) - rate - torch.lgamma(k + 1)


def calculateAllLogProbs(expression, numMean, numStd, stepSize, stdList, includeUniform=False):

    #TODO: add binomial prob in addition to the Gaussian prob!
    #TODO: or replace Gaussian with negative binomial 

    #calculateAllLogProbs(expression, numMean, numStd, stepSize)

    logProbsAll = torch.zeros(( expression.shape[0], expression.shape[1], numMean, numStd  ))
    if includeUniform:
        logProbsAll_U = torch.zeros(( expression.shape[0], expression.shape[1], numMean, numStd  ))

    for a in range(numStd):
        for b in range(numMean):
            mean1 = float(b) * stepSize
            
            dist = torch.distributions.Normal(loc= mean1, scale=stdList[a])
            log_probs = dist.log_prob(expression)

            #log_probs = negbin_log_prob(torch.exp2(expression), np.exp2(mean1), np.exp2(mean1) * stdList[a], eps=1e-8)

            logProbsAll[:, :, b, a] = log_probs
            if includeUniform:
                logProbsAll_U[:, :, b, a] = log_probs + np.log(stdList[a])

    #logProbsAll = logProbsAll.reshape(( expression.shape[0], expression.shape[1], numStd*numMean ))
    #if includeUniform:
    #    logProbsAll_U = logProbsAll_U.reshape(logProbsAll.shape)
    #    return logProbsAll, logProbsAll_U
    #else:
    return logProbsAll




def getCrossProbSamples(probabilities, samples):

    #print (probabilities.shape)
    #print (samples.shape)

    argAll = np.argwhere( np.zeros(samples.shape) > -1 )
    crossProb_samples = probabilities[:, argAll[:, 1],  samples[argAll[:, 0],  argAll[:, 1] ]  ]
    crossProb_samples = crossProb_samples.reshape(( crossProb_samples.shape[0], samples.shape[0], samples.shape[1] ))
    crossProb_samples = torch.sum(crossProb_samples, axis=2)

    return crossProb_samples





def partial_getCrossProbSamples(probabilities, samples, crossSize):

    otherSize = samples.shape[0] // crossSize

    probabilities = probabilities.reshape(( crossSize, probabilities.shape[0] // crossSize, probabilities.shape[1], probabilities.shape[2] ))
    samples = samples.reshape((  crossSize, samples.shape[0] // crossSize, samples.shape[1] ))


    argAll = np.argwhere( np.zeros(samples.shape) > -1 )
    crossProb_samples = probabilities[:, argAll[:, 1],  argAll[:, 2], samples[argAll[:, 0],  argAll[:, 1], argAll[:, 2] ]  ]
    crossProb_samples = crossProb_samples.reshape(( crossProb_samples.shape[0], samples.shape[0], samples.shape[1], samples.shape[2] ))
    crossProb_samples = torch.sum(crossProb_samples, axis=3)
    #crossProb_samples = crossProb_samples.reshape(( crossProb_samples.shape[0], crossProb_samples.shape[1]*crossProb_samples.shape[2] ))


    #print (crossProb_samples.shape)
    crossProb_samples_paste = torch.zeros( (  crossSize, otherSize, crossSize, otherSize ) )
    crossProb_samples_paste[:] = -1e30
    #crossProb_samples_paste = crossProb_samples_paste

    idx = torch.arange(otherSize)
    crossProb_samples_paste[:, idx, :, idx] = torch.moveaxis(crossProb_samples[:, :, idx], -1, 0)

    crossProb_samples_paste = crossProb_samples_paste.reshape(( crossSize*otherSize, crossSize*otherSize ))

    return crossProb_samples_paste

def partial_getCrossProbSamples_onehot(probabilities, samples_oneHot, crossSize, fill_value=-1e30, cumProb=False ):
    """
    probabilities:   (N, L, C)
    samples_oneHot:  (N, L, C)
    crossSize: int, where N = crossSize * otherSize

    Returns:
        crossProb_samples_paste: (N, N)
    """

    N, L, C = probabilities.shape
    N2 = samples_oneHot.shape[0]
    dupGen = N2 // N
    otherSize = N // crossSize
    #otherSize2 = N2 // crossSize
    crossSize2 = crossSize * dupGen
    #print (probabilities.shape)
    #print (samples_oneHot.shape)
    #quit()

    # reshape into (crossSize, otherSize, L, C)
    probabilities = probabilities.reshape(crossSize, otherSize, L, C)
    samples_oneHot = samples_oneHot.reshape(crossSize2, otherSize, L, C)

    # For each fixed "otherSize" index b, compute:
    # out[k, a, b] = sum_{l,c} probabilities[k,b,l,c] * samples_oneHot[a,b,l,c]
    #
    # Result shape: (crossSize, crossSize, otherSize)

    #print (probabilities.shape, samples_oneHot.shape)
    #print (N, L, C)
    #quit()

    if cumProb:
        crossProb_samples = torch.einsum('kblc,ablc->kabl', probabilities, samples_oneHot)


        #L
        # Build the pasted matrix on the SAME device/dtype
        crossProb_samples_paste = torch.full(
            (crossSize, otherSize, crossSize2, otherSize, L),
            fill_value,
            device=probabilities.device,
            dtype=probabilities.dtype
        )

        idx = torch.arange(otherSize, device=probabilities.device)

        crossProb_samples_paste[:, idx, :, idx] = torch.moveaxis(crossProb_samples[:, :, idx], 2, 0)


        crossProb_samples_paste = crossProb_samples_paste.reshape(crossSize * otherSize, crossSize2 * otherSize, L)
        #crossProb_samples_paste = torch.cumsum(crossProb_samples_paste, axis=2)
        

    else:
        crossProb_samples = torch.einsum('kblc,ablc->kab', probabilities, samples_oneHot)
    
        # Build the pasted matrix on the SAME device/dtype
        crossProb_samples_paste = torch.full(
            (crossSize, otherSize, crossSize2, otherSize),
            fill_value,
            device=probabilities.device,
            dtype=probabilities.dtype
        )

        idx = torch.arange(otherSize, device=probabilities.device)
        crossProb_samples_paste[:, idx, :, idx] = torch.moveaxis(crossProb_samples[:, :, idx], -1, 0)

        crossProb_samples_paste = crossProb_samples_paste.reshape(crossSize * otherSize, crossSize2 * otherSize)

    return crossProb_samples_paste



def getObservationProbs(allExpressionProbs_batch, samples_oneHot, offPolicyProb_samples, onPolicyProb_samples, dupGen):

    

    timeList = []
    timeList.append(time.time())
    dataProb_samples = getProbSamples_onehot(allExpressionProbs_batch, samples_oneHot)

    timeList.append(time.time())
    importance_log = (onPolicyProb_samples - offPolicyProb_samples).detach()


    if False:
        normalizer = torch.logsumexp(importance_log, axis=0) - np.log(importance_log.shape[0])
        importance_log = importance_log - normalizer

    #importance_log = importance_log.reshape((1, -1))

    probXS = dataProb_samples + importance_log
    probXS = probXS.reshape((dupGen, probXS.shape[0] // dupGen ))

    probX = torch.logsumexp(probXS, axis=0)
    rewards = probXS.detach() - probX.reshape((1, -1)).detach()
    rewards = torch.exp(rewards)
    rewards = rewards.reshape((-1,))


    return probX, rewards



def giveOnPolicyRewards(offPolicyProb_samples, onPolicyProb_samples, dataProb_samples):

    

    

    

    return rewards


def giveOffPolicyRewards(offPolicyProb, onPolicyProb, allExpressionProbs):

    True

    print (offPolicyProb.shape)
    print (onPolicyProb.shape)
    print (allExpressionProbs.shape)
    quit()



def findOptimalCoverage(offPolicyProb, allExpressionProbs_batch, numMean):

    allExpressionProbs_batch = allExpressionProbs_batch.reshape(( offPolicyProb.shape[0], offPolicyProb.shape[1], offPolicyProb.shape[2] // numMean, numMean ))
    offPolicyProb = offPolicyProb.reshape(( offPolicyProb.shape[0], offPolicyProb.shape[1], offPolicyProb.shape[2] // numMean, numMean ))


    numShiftTry = 10

    shiftScores = np.zeros((  offPolicyProb.shape[0], numShiftTry ))

    for shiftIndex in range(numShiftTry):

        print (offPolicyProb[0, 0])

        offPolicyProb_mod = np.copy(offPolicyProb[ (np.arange(numMean) + shiftIndex) % numShiftTry ])

        print (offPolicyProb_mod[0, 0])
        quit()


    

    print (offPolicyProb.shape)
    print (allExpressionProbs_batch.shape)
    quit()



def gumbel_sample(logits_clone):

    if len(logits_clone.shape) == 2:

        u = torch.rand_like(logits_clone)                  # uniform(0,1)
        gumbel = -torch.log(-torch.log(u))                 # Gumbel(0,1)
        actions_pi_prime = (logits_clone + gumbel).argmax(dim=1)
        return actions_pi_prime
    
    if len(logits_clone.shape) == 3:
        shape1 = logits_clone.shape
        logits_clone = logits_clone.reshape(( shape1[0]*shape1[1], shape1[2] ))
        actions_pi_prime = gumbel_sample(logits_clone)
        actions_pi_prime = actions_pi_prime.reshape(( shape1[0], shape1[1] ))
        return actions_pi_prime
    






def negbin_log_prob(x, mean, std, eps=1e-5, noNoise=False):

    #print (x.shape, mean.shape, std.shape)

    
    #if not torch.is_tensor(mean):
    #    mean = torch.tensor(mean, dtype=torch.float32, device=x.device)
    #if not torch.is_tensor(std):
    #    std = torch.tensor(std, dtype=torch.float32, device=x.device)

    #mean = mean.to(x.device)
    #std = std.to(x.device)

    timeList = []
    timeList.append(time.time())

    bio_var = std**2
    #if noNoise:
    #    total_var = mean + eps 
    #else:
    #    total_var = mean + bio_var

    timeList.append(time.time())

    r = mean**2 / (bio_var + eps)

    timeList.append(time.time())

    p = r / (r + mean)

    timeList.append(time.time())

    p = torch.clamp(p, eps, 1 - eps)

    timeList.append(time.time())

    log_prob = (
        torch.lgamma(x + r)
        - torch.lgamma(r)
        - torch.lgamma(x + 1)
        + r * torch.log(p)
        + x * torch.log1p(-p)
    )

    timeList.append(time.time())

    timeList = np.array(timeList)
    #print (timeList[1:] - timeList[:-1])
    #quit()

    return log_prob
    
def doGaussian(values, mean, stdFactor, scales, noNoise=False):

    device = values.device

    timeList = []

    timeList.append(time.time())
    
    stdFactor = stdFactor.reshape((1, stdFactor.shape[0], stdFactor.shape[1] ))
    mean = mean.reshape(( 1, 1, mean.shape[0] ))
    values = values.reshape(( values.shape[0], values.shape[1], 1 ))

    timeList.append(time.time())

    #print (stdFactor.shape)
    #rint (mean.shape)
    #print (values.shape)
    #print ("A")
    #quit()
    
    #stdFactor = torch.sigmoid(stdFactor * 10.0) + 0.25 #Good, used for offPol_CAll_specialSTD11
    stdFactor = torch.sigmoid((stdFactor * 10.0)) * 1.5

    #stdFactor = torch.sigmoid((stdFactor * 10.0)) * 10.0

    timeList.append(time.time())

    
    scales_adj = scales - torch.median(scales)
    #print (scales_adj[0])
    scales_adj = scales_adj.reshape((scales_adj.shape[0], 1, 1))

    timeList.append(time.time())
    
    #print (mean[0, 0, 0], scales_adj[0, 0, 0])
    #print (mean.device , scales_adj.device)
    mean2 = torch.exp2(mean + scales_adj)

    #Added Tue Apr 28
    #mean2 = mean2 - 0.9
    #mean2[mean2 < 0.1] = 0.1

    mean2 = mean2 - 0.99
    mean2[mean2 < 0.01] = 0.01


    #print (mean2.shape)
    #print (mean2[:30, 0, 0].cpu().data.numpy())
    #print (mean2[:30, 0, 1].cpu().data.numpy())
    #print (values[:30, 1768, 0 ].cpu().data.numpy())
    #quit()


    #print (stdFactor.shape, mean2.shape)
    #quit()
    stdFactor = stdFactor * mean2
    #values2 = torch.exp2(values)
    #values2 = (values2 - mean2) / stdFactor
    
    #torch.mps.synchronize()
    timeList.append(time.time())

    if False:
        values2 = (values - mean2) / stdFactor

        logProb = -0.5 * (values2 ** 2.0)
        logProb = logProb - torch.log(stdFactor)
        logProb = logProb - (np.log(  np.pi * 2) / 2)
    

    #zeros = torch.zeros((  values.shape[0], stdFactor.shape[1], stdFactor.shape[2] )).to(device)

    time1 = time.time()
    if noNoise:
        #logProb = poisson_log_prob(values+zeros, mean2+zeros, eps=1e-8)
        logProb = poisson_log_prob(values, mean2)
    else:

        #logProb = negbin_log_prob(values+zeros, mean2+zeros, stdFactor+zeros, eps=1e-8, noNoise=noNoise)
        logProb = negbin_log_prob(values, mean2, stdFactor, noNoise=noNoise)

    #torch.mps.synchronize()
    timeList.append(time.time())
    

    #timeList = np.array(timeList)
    #print (timeList[1:] - timeList[:-1])
    #quit()

    #print ('NegB', time.time() - time1)
    #print (negbin_log_prob(  (values+zeros)[:1, 0, 3], (mean2+zeros)[:1, 0, 3], (stdFactor+zeros)[:1, 0, 3]  , eps=1e-8))

    #print ((values+zeros)[:1, 0, 3], (mean2+zeros)[:1, 0, 3], (stdFactor+zeros)[:1, 0, 3] )


    #sns.heatmap(logProb[0].data.numpy())
    #plt.show()

    #print (logProb[0, 0, 3])
    #print (values[0, 0])
    #print (mean2[0, 0, 3])
    #print (stdFactor[0, 0, 3])

    #quit()

    

    return logProb



def doGaussian_better(values, mean, stdFactor, scales, noNoise=False, eps=1e-5):  # eps=1e-8):
    """
    values:    (A, B)
    mean:      (C,)
    stdFactor: (B, C)
    scales:    (A,)
    returns:   (A, B, C)
    """

    A, B = values.shape
    C = mean.shape[0]
    device = values.device
    dtype = values.dtype

    # Small-tensor work
    scales_adj = scales - torch.median(scales)              # (A,)
    s = torch.exp2(scales_adj)                              # (A,)
    m = torch.exp2(mean)                                    # (C,)
    #sf = torch.sigmoid(stdFactor * 10.0) * 1.5              # (B, C)
    #sf = torch.sigmoid( (stdFactor - 0.5) * 10.0) * 1.0
    sf = torch.sigmoid( (stdFactor - 1.0) * 10.0) * 1.0


    sns.heatmap(sf.cpu().data.numpy())
    plt.show()

    print (sf.shape)
    quit()

    # mean2[a,c]
    mu_ac = s[:, None] * m[None, :]                         # (A, C)

    if noNoise:
        # Materialize once before heavy kernel work
        x_full = values[:, :, None].expand(A, B, C).contiguous()
        mu_full = mu_ac[:, None, :].expand(A, B, C).contiguous()
        return poisson_log_prob(x_full, mu_full, eps=eps)

    # r[b,c]
    r_bc = 1.0 / (sf * sf + eps)                            # (B, C)

    # Materialize once
    x_full  = values[:, :, None].expand(A, B, C).contiguous()
    mu_full = mu_ac[:, None, :].expand(A, B, C).contiguous()
    r_full  = r_bc[None, :, :].expand(A, B, C).contiguous()

    p = r_full / (r_full + mu_full)
    p = torch.clamp(p, eps, 1 - eps)


    final =  (
        torch.lgamma(x_full + r_full)
        - torch.lgamma(r_full)
        - torch.lgamma(x_full + 1)
        + r_full * torch.log(p)
        + x_full * torch.log1p(-p)
    )

    #print (torch.mean(torch.lgamma(x_full + r_full)))
    #print (torch.mean(torch.lgamma(r_full)))
    #print(torch.mean( torch.lgamma(x_full + 1) )) 

    #print (-p)
    #print (torch.mean(torch.log1p(-p)))


        #+ r_full * torch.log(p)
        #+ x_full * torch.log1p(-p))

    #print (final[0, 0, 0])
    #print (torch.mean(final[0]))
    #quit()

    return final




def largeAdvanced_getObservationProbs(allExpressionProbs_batch, samples_oneHot, offPolicyProb_samples, onPolicyProb_samples, offPolicyProb, onPolicyProb, dupGen):

    

    timeList = []
    timeList.append(time.time())

    device = allExpressionProbs_batch.device

    allExpressionProbs_dup = allExpressionProbs_batch.repeat(dupGen, 1, 1)

    dataProb_samples = getProbSamples_onehot(allExpressionProbs_dup, samples_oneHot, cumProb=True)


    importance_log = (onPolicyProb_samples - offPolicyProb_samples).detach()
    probXS = importance_log + dataProb_samples
    probXS = torch.cumsum(probXS, axis=1)
    zero1 = torch.zeros((probXS.shape[0], 1)).to(device)
    probXS_mod = torch.cat( [  zero1   ,   probXS[:, :-1]  ], axis=1 )
    probXS_mod = probXS_mod.reshape((probXS_mod.shape[0], probXS_mod.shape[1], 1))

    
    probXS_mod = probXS_mod + onPolicyProb + allExpressionProbs_dup

    probXS_mod = probXS_mod.reshape(( dupGen, probXS_mod.shape[0] // dupGen, probXS_mod.shape[1], probXS_mod.shape[2] ))
    probX_mod = torch.logsumexp(probXS_mod, axis=(0, 3))
    probXS_mod = probXS_mod - probX_mod.reshape((1, probX_mod.shape[0], probX_mod.shape[1], 1))

    rewards = torch.exp(probXS_mod)
    rewards = rewards.reshape(( rewards.shape[0]*rewards.shape[1], rewards.shape[2], rewards.shape[3] ))
    
    probX = probX_mod[:, -1]
    

    return probX, rewards






def advanced_getObservationProbs(allExpressionProbs_batch, samples_oneHot, offPolicyProb_samples, onPolicyProb_samples, crossSize):

    #print (allExpressionProbs_batch.shape, samples_oneHot.shape, offPolicyProb_samples.shape, onPolicyProb_samples.shape, offPolicyProb.shape, onPolicyProb.shape, crossSize)


    timeList = []
    timeList.append(time.time())

    device = allExpressionProbs_batch.device

    dataProb_samples = partial_getCrossProbSamples_onehot(allExpressionProbs_batch, samples_oneHot, crossSize, cumProb=True)



    timeList.append(time.time())

    importance_log = (onPolicyProb_samples - offPolicyProb_samples).detach()
    importance_log = importance_log.reshape(( 1, importance_log.shape[0], importance_log.shape[1] ))   

    probXS = importance_log + dataProb_samples

    probXS = torch.cumsum(probXS, axis=2)

    probX = torch.logsumexp(probXS, axis=1)
    probXS = probXS - probX.reshape((probX.shape[0], 1, probX.shape[1]))

    rewards = torch.logsumexp(probXS, axis=0) - np.log(probXS.shape[1])
    rewards = torch.exp(rewards)


    #print (rewards.shape)
    #print (torch.sum(rewards))
    #quit()

    probX = probX[:, -1]
    

    return probX, rewards




def inverse_cumsum(y, dim=0):
    first = y.narrow(dim, 0, 1)
    diff = y.diff(dim=dim)
    return torch.cat([first, diff], dim=dim)





    #ataProb_samples = partial_getCrossProbSamples_onehot(allExpressionProbs_batch, samples_oneHot, crossSize, cumProb=True)
    #mportance_log = (onPolicyProb_samples - offPolicyProb_samples).detach()
    #importance_log = importance_log.reshape(( 1, importance_log.shape[0], importance_log.shape[1] ))   
    #robXS = importance_log + dataProb_samples
    #probXS = torch.cumsum(probXS, axis=2)
    #probX = torch.logsumexp(probXS, axis=0)



    #dataProb_samples = partial_getCrossProbSamples_onehot(allExpressionProbs_batch, samples_oneHot, crossSize, cumProb=cumProb)
    #importance_log = (onPolicyProb_samples - offPolicyProb_samples).detach()
    #importance_log = importance_log.reshape((1, -1))
    #probXS = importance_log + dataProb_samples
    #probX = torch.logsumexp(probXS, axis=1)





    


def doSearch(autoregModel, allExpressionProbs, samples, geneOrder, maxExpression, device):

    #device = autoregModel.device()

    samplesDone = np.zeros(samples.shape[0], dtype=int)


    lastUpdates = np.ones(( samples.shape[0], samples.shape[1] ), dtype=int)

    

    #while 0 in samplesDone:

    while np.max(np.sum(lastUpdates, axis=1)) >= 1:
        samples_copy = samples.clone()
        #updated = np.zeros(samples.shape[0], dtype=int)
        for geneIndex0 in range(allExpressionProbs.shape[1]):
            geneIndex = geneOrder[geneIndex0]

            samples_copy1 = samples.clone()

            print (geneIndex0)

            maxExpression_now = maxExpression[geneIndex] + 1

            print ('max', maxExpression_now)

            timeList = []
            torch.mps.synchronize()
            timeList.append(time.time())


            #argNotDone = np.argwhere(samplesDone == 0)[:, 0]
            argNotDone = np.argwhere(np.sum(lastUpdates, axis=1) >= 1)[:, 0]

            if argNotDone.shape[0] >= 1:
            
                samples_now = samples[argNotDone].clone()



                _, inverse1 = np.unique(samples_now.cpu().data.numpy(), axis=0, return_inverse=True)
                #inverse1 = uniqueValMaker(samples_now)
                _, index1 = np.unique(inverse1, return_index=True)
                samples_now = samples_now[index1]
                print ('inv2', np.unique(inverse1).shape, inverse1.shape)


                samples_now = samples_now.reshape((samples_now.shape[0], samples_now.shape[1], 1))
                #samples_now = samples_now[:, :, torch.zeros(allExpressionProbs.shape[2], dtype=int) ]
                samples_now = samples_now[:, :, torch.zeros(maxExpression_now, dtype=int) ]

                torch.mps.synchronize()
                timeList.append(time.time()) #1

                samples_now[:, geneIndex, torch.arange(maxExpression_now).to(device)  ] = torch.arange(maxExpression_now).to(device)
                samples_now = torch.swapaxes(samples_now, 1, 2)
                optionNum = samples_now.shape[1]
                shape1 = samples_now.shape

                samples_now_input = samples_now.reshape((samples_now.shape[0]*samples_now.shape[1], samples_now.shape[2])) 
                samples_now = samples_now[inverse1].reshape((inverse1.shape[0]*samples_now.shape[1], samples_now.shape[2])) 

                #print (samples_now_input.shape)
                #print (samples_now.shape)
                #quit()

                #samples_now_input = samples_now[:, ].reshape((samples_now.shape[0]*samples_now.shape[1], samples_now.shape[2])) 

                torch.mps.synchronize()
                timeList.append(time.time()) #2
                

                #onPolicyProb = autoregModel(torch.tensor(samples_now_input).long().to(device))
                onPolicyProb = autoregModel(samples_now_input)

                torch.mps.synchronize()
                timeList.append(time.time()) #3

                numDist = onPolicyProb.shape[2]

                onPolicyProb = onPolicyProb.reshape(( shape1[0], shape1[1], onPolicyProb.shape[1], onPolicyProb.shape[2] ))
                onPolicyProb = onPolicyProb[inverse1]
                onPolicyProb = onPolicyProb.reshape(( onPolicyProb.shape[0]*onPolicyProb.shape[1], onPolicyProb.shape[2], onPolicyProb.shape[3] ))
                
                #print (onPolicyProb.shape, shape1)
                #quit()
                torch.mps.synchronize()
                timeList.append(time.time()) #4


                #time2 = time.time()

                samples_oneHot = F.one_hot(samples_now, num_classes=numMean).float()

                #onPolicyProb_samples = getProbSamples(onPolicyProb, samples_now)
                onPolicyProb_samples = getProbSamples_onehot(onPolicyProb, samples_oneHot)

                #print (time.time() - time2)

                torch.mps.synchronize()
                timeList.append(time.time()) #5


                #samples_now = torch.tensor(samples_now).long()
                #time2 = time.time()
                #oneHot1 = sampleToOneHot(samples_now, numDist)
                #oneHot = nn.Embedding(numDist, 1)(samples_now)

                #print (oneHot.shape)
                #quit()

                #print (torch.mean(torch.abs(oneHot-oneHot1)))
                
                #print (time.time() - time2)

                #time2 = time.time()
                #onPolicyProb_samples = torch.sum(onPolicyProb * oneHot, axis=(1, 2))
                #print (time.time() - time2)

                #quit()

                dataProbNow = allExpressionProbs[argNotDone]
                dataProbNow = dataProbNow.reshape(( dataProbNow.shape[0], 1, dataProbNow.shape[1], dataProbNow.shape[2] ))
                dataProbNow = dataProbNow[:,  np.zeros(optionNum, dtype=int) ]
                dataProbNow = dataProbNow.reshape(( dataProbNow.shape[0]*dataProbNow.shape[1], dataProbNow.shape[2], dataProbNow.shape[3] ))

                torch.mps.synchronize()
                timeList.append(time.time()) #6

                #print (dataProbNow.shape, samples_now.shape)
                
                dataProb_samples = getProbSamples_onehot(dataProbNow, samples_oneHot)


                torch.mps.synchronize()
                timeList.append(time.time()) #7

                probAdjusted = dataProb_samples + onPolicyProb_samples
                #probAdjusted = probAdjusted.data.numpy()


                probAdjusted = probAdjusted.reshape(( argNotDone.shape[0], optionNum ))
                bestSelection = torch.argmax(probAdjusted, axis=1)

                samples_now = samples_now.reshape(( argNotDone.shape[0], optionNum, allExpressionProbs.shape[1] ))
                samples_new = samples_now[np.arange(bestSelection.shape[0]), bestSelection]

                samples[argNotDone] = samples_new



                torch.mps.synchronize()
                timeList.append(time.time()) #8

                timeList = np.array(timeList)
                timeList = timeList[1:] - timeList[:-1]
                #print (np.round(timeList, decimals=2))
                print (np.sum(timeList))
                #quit()

                #[0.01 0.   0.01 0.26 0.01 0.18 0.   0.1 ]
                #[0.   0.01 0.47 0.02 0.07 0.01 0.03 0.  ]

                
                error2 = torch.abs(samples[:, geneIndex] - samples_copy1[:, geneIndex])
                error2 = error2.cpu().data.numpy()
                lastUpdates[error2 == 0, geneIndex] = 0
                lastUpdates[error2 >= 1, geneIndex] = 1




        error = np.sum(np.abs(samples.cpu().data.numpy() - samples_copy.cpu().data.numpy()), axis=1)

        samplesDone[error == 0] = 1

        print ('samplesDone', np.sum(samplesDone))
        #print (error.shape)
        #quit()

    return samples




def doInference(offModelFile, onModelFile, expression, meanFactor, scales):


    device = torch.device("cpu")
    #device = torch.device("mps")

    noNoise = False
    #noNoise = True


    expression = expression.to(device)

    with torch.no_grad():

        #expression = expression[:100]
        #expression = expression[:50]

        #expression = expression[:50]
        expression = expression[:200]
        #expression = expression[:500]
        #expression = expression[:2]
        #expression = expression[:1000]


        scales = scales[:expression.shape[0]].to(device)
        encodeModel = torch.load(offModelFile).to(device)
        autoregModel = torch.load(onModelFile).to(device)

        #allExpressionProbs = calculateAllLogProbs(expression, numMean, numStd, stepSize, stdList)

        #stdFactor = autoregModel.stdFactor * 10.0
        #stdFactor = torch.log_softmax(stdFactor, axis=2)
        #allExpressionProbs = torch.logsumexp(allExpressionProbs + stdFactor.reshape((1, stdFactor.shape[0], stdFactor.shape[1], stdFactor.shape[2])), axis=3)
        stdFactor = autoregModel.stdFactor[:, :, 0]
        #####meanFactor = torch.arange(numMean) * stepSize
        meanFactor = meanFactor.to(device)
        #print ("Hi")
        #print (expression.shape, meanFactor.shape, stdFactor.shape, scales.shape)
        allExpressionProbs = doGaussian(expression, meanFactor, stdFactor, scales, noNoise=noNoise)

        bestProb = torch.argmax(allExpressionProbs, axis=2 )
        maxExpression, _ = torch.max(bestProb, axis=0)

        #plt.plot(maxExpression.cpu().data.numpy())
        #plt.show()
        #quit()
        

        #print (expression[:, 1768])
        #print (allExpressionProbs[:, 1768, 0])
        #print (allExpressionProbs[:, 1768, 1])
        #quit() #1768
        

        

        offPolicyProb = encodeModel(torch.log2(expression+1))  
        #offPolicyProb = encodeModel(expression)

        #for a in range(100):
        #    img1 = np.exp(torch.log_softmax(offPolicyProb, axis=2)[a].cpu().data.numpy())
        #    img1[img1 > 0.1] = 0.1
        #    sns.heatmap(img1)
        #    plt.show()

        #quit()

        offPolicyProb = offPolicyProb + allExpressionProbs
        offPolicyProb = torch.log_softmax(offPolicyProb, axis=2)

        #print (offPolicyProb.shape)

        #sns.heatmap(np.exp(offPolicyProb[0].cpu().data.numpy()))
        #plt.show()
        #uit()


        offPolicyProb_np = np.exp(offPolicyProb.cpu().data.numpy())
        offPolicyProb_np = np.sort(offPolicyProb_np, axis=2)[:, :, -1::-1]
        offPolicyProb_np = np.mean(offPolicyProb_np, axis=0)

        geneOrder = np.argsort(offPolicyProb_np[:, 0])
        ###geneOrder = np.argsort(offPolicyProb_np[:, 0] * -1)
        #offPolicyProb_np = offPolicyProb_np[np.argsort(offPolicyProb_np[:, 0])]
        #print (offPolicyProb_np.shape)
        #quit()

        #sns.heatmap(  offPolicyProb_np )
        #plt.show()
        #quit()

        dist = torch.distributions.Categorical(logits=offPolicyProb)
        for sampleDup in range(100):# range(100):# range(100):
            print ('sampleDup', sampleDup)

            if sampleDup == 0:
                samples = torch.argmax(offPolicyProb, axis=2)
            else:
                samples = dist.sample() 

            samples_oneHot = F.one_hot(samples, num_classes=numMean).float()

            crossSize=samples.shape[0]

            #dataProb_samples = getCrossProbSamples(allExpressionProbs, samples)
            dataProb_samples = partial_getCrossProbSamples_onehot(allExpressionProbs, samples_oneHot, crossSize)
            onPolicyProb = autoregModel(samples)
            
            
            onPolicyProb_samples = getProbSamples_onehot(onPolicyProb, samples_oneHot)

            probAdjusted = dataProb_samples + onPolicyProb_samples.reshape((1, -1))
            probAdjusted = probAdjusted.cpu().data.numpy()



            samples = samples.cpu().data.numpy()
            if sampleDup == 0:
                samples_all = np.copy(samples)
                probAdjusted_all = np.copy(probAdjusted)
            else:
                samples_all = np.concatenate(( samples_all, samples ), axis=0)
                probAdjusted_all = np.concatenate(( probAdjusted_all,probAdjusted  ), axis=1)

        #print (samples_all.shape)

        
        #quit()

        inverse1 = uniqueValMaker(samples_all)
        print ('inv1', np.unique(inverse1).shape, inverse1.shape)

        bestChoice = np.argmax( probAdjusted_all , axis=1  )

        samples_choice = samples_all[bestChoice]

        np.savez_compressed('./data/RNA/pred/pred200_PBMC_top2000_simple_noNoise.npz', samples_choice)
        quit()

        samples_choice = torch.tensor(samples_choice).to(device)

        samples_choice = doSearch(autoregModel, allExpressionProbs, samples_choice, geneOrder, maxExpression, device)

        samples_choice = samples_choice.cpu().data.numpy()

        #np.savez_compressed('./data/RNA/pred/1000Pred.npz', samples_choice)
        #np.savez_compressed('./data/RNA/pred/100Pred_train_noNoise.npz', samples_choice)
        #np.savez_compressed('./data/RNA/pred/AllPred_N0001.npz', samples_choice)
        np.savez_compressed('./data/RNA/pred/pred200_PBMC_top2000_noNoise.npz', samples_choice)


        inverse1 = uniqueValMaker(samples_choice)
        print ('inv2', np.unique(inverse1).shape, inverse1.shape)

        if False:
            for a in range(100):
                xPlot = samples_choice[:, a].astype(float) % 10
                xPlot = xPlot + (np.random.random(size=xPlot.shape[0]) * 0.05)
                plt.scatter(xPlot , expression[:, a])
                plt.show()
        #quit()



        #for a in range(100):
        #   plt.plot(expression[a])
        #    plt.plot(expression[a+1])
        #    plt.plot(samples_choice[a] % 10)
        #    plt.plot(samples_choice[a+1] % 10)
        #    plt.show()
        #quit()
        print (samples_choice[0])

        
        print (samples_choice.shape)
    

def get_clustermap_order(data, **clustermap_kwargs):
    """
    Runs seaborn clustermap once and returns row/col ordering.
    """
    cg = sns.clustermap(data, **clustermap_kwargs)

    #row_order = cg.dendrogram_row.reordered_ind
    #col_order = cg.dendrogram_col.reordered_ind

    row_order = np.asarray(cg.dendrogram_row.reordered_ind, dtype=int)
    col_order = np.asarray(cg.dendrogram_col.reordered_ind, dtype=int)

    plt.close()  # prevent showing the figure
    return row_order, col_order


def plot_with_order(data, row_order, col_order,  cellTypeIndicator=[], **heatmap_kwargs):
    """
    Plot a heatmap using a fixed ordering.
    """
    ordered = data[np.ix_(row_order, col_order)]

    if len(cellTypeIndicator) > 0:
        cellTypeIndicator = cellTypeIndicator[0]

        width = col_order.shape[0] // 20
        cellTypeIndicator_vis = cellTypeIndicator.reshape((-1, 1))[:, np.zeros(width, dtype=int) ] * np.max(ordered)


        ordered = np.concatenate(( ordered, cellTypeIndicator_vis ), axis=1)

    sns.heatmap(ordered, **heatmap_kwargs)
    plt.xlabel('gene')
    plt.ylabel('cell')
    plt.tight_layout()
    plt.show()



def analyzeInference(expression):


    from scipy.stats import ttest_ind

    expression = expression.data.numpy()

    data_path = "./data/RNA/raw/"
    #samples_choice = loadnpz('./data/RNA/pred/50Pred_14.npz') # % 20

    #samples_choice = loadnpz('./data/RNA/pred/500Pred_nb.npz')

    #samples_choice = loadnpz('./data/RNA/pred/500Pred_nb_12.npz')
    #samples_choice = loadnpz('./data/RNA/pred/500Pred_nb_12.npz')
    #samples_choice = loadnpz('./data/RNA/pred/500Pred.npz')

    #samples_choice = loadnpz('./data/RNA/pred/pred50_PBMC_allGenes_simpleBig.npz')
    samples_choice = loadnpz('./data/RNA/pred/pred500_PBMC_top100.npz')
    #samples_choice = loadnpz('./data/RNA/pred/pred50_PBMC_allGenes.npz')

    #protein = pd.read_csv(data_path + "GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv", index_col=0).T
    #protein = protein.to_numpy()


    

    #expression = loadnpz('./data/RNA/input/PBMC_Tcell_top500.npz', allow_pickle=True)
    #expression = loadnpz('./data/RNA/input/PBMC_Tcell.npz', allow_pickle=True)
    scales = loadnpz('./data/RNA/input/PBMC_Tcell_scales.npz')


    #topGenes = loadnpz('./data/RNA/input/PBMC_topVar100.npz')
    #expression = expression[:, topGenes]

    

    #expression = loadnpz('./data/RNA/input/expression.npz')

    #scales = loadnpz('./data/RNA/input/scales.npz')
    scales = scales - np.median(scales)
    expression = expression * np.exp2(-scales).reshape((-1, 1))
    expression_trans = np.log2(expression + 1)
    #expression_trans = np.log2(expression + 2)

    

    np.random.seed(0)
    perm0 = np.random.permutation(expression.shape[0])
    #expression_trans = expression_trans[perm0]
    expression_trans = expression_trans[:samples_choice.shape[0]]
    #expression_trans = np.log2(expression.astype(float) + 1)
    #expression_trans = torch.tensor(expression_trans).float()

    expression_trans[expression_trans>9.5] = 9.5 #For plotting scale

    cellType1 = loadnpz('./data/RNA/input/PBMC_Tcell_top1000_cellType1.npz', allow_pickle=True)
    #cellType1 = loadnpz('./data/RNA/input/PBMC_Tcell_top1000_cellType1.npz', allow_pickle=True)
    cellType1 = cellType1[perm0][:samples_choice.shape[0]]

    #protein = protein[perm0][:samples_choice.shape[0]]

    #scVI_vals = loadnpz('./data/RNA/scVI/denoised2.npz')
    #scVI_vals = scVI_vals[perm0][:samples_choice.shape[0]]
    #scVI_log = np.log2(scVI_vals+1)
    
    

    #sns.clustermap(samples_choice // 20)
    #plt.show()
    #quit()

    print (expression_trans.shape)

    #row_order, col_order = get_clustermap_order(samples_choice)
    #row_order, col_order = get_clustermap_order(expression_trans)
    #row_order, col_order = row_order.astype(int), col_order.astype(int)

    #col_order = np.argsort(np.std(samples_choice, axis=0))
    col_order = np.argsort(np.std(expression_trans, axis=0))
    row_order = np.argsort(cellType1)

    print (np.max(col_order))
    print (np.max(row_order))
    print (expression_trans.shape)
    print (samples_choice.shape)

    ##########np.savez_compressed('./data/RNA/input/PBMC_topVar.npz', col_order )
    #quit()
    

    scales = loadnpz('./data/RNA/input/PBMC_Tcell_scales.npz')

    print (cellType1[row_order])

    _, cellType_inverse = np.unique(cellType1[row_order], return_inverse=True)
    _, cellTypeIndex = np.unique(cellType_inverse, return_index=True)
    cellType_inverse = cellTypeIndex[cellType_inverse]
    _, cellType_inverse = np.unique(cellType_inverse, return_inverse=True)
    cellTypeIndicator = cellType_inverse % 2

    print (cellType1[cellTypeIndex])

    col_order = col_order[-100:]


    #meanValues1 = expression_trans[row_order][100:300][:, col_order]
    #meanValues2 = expression_trans[row_order][350:500][:, col_order]
    #meanValues1 = np.mean(meanValues1.data.numpy(), axis=0)
    #meanValues2 = np.mean(meanValues2.data.numpy() + 0.8, axis=0)
    

    #plt.plot(meanValues1)
    #plt.plot(meanValues2)
    #lt.show()

    #print (row_order)
    #quit()
    
    #quit()

    #sns.clustermap(expression_trans[row_order][:, col_order[510:] ])
    #plt.show()
    #quit()

    #plt.plot(np.mean(expression_trans[4:22][:, col_order[480:]], axis=0) )
    #plt.plot(np.mean(expression_trans[30:][:, col_order[480:]], axis=0) )
    #plt.plot(np.mean(samples_choice[4:22][:, col_order[480:]] / 4, axis=0) )
    #plt.plot(np.mean(samples_choice[30:][:, col_order[480:]] / 4, axis=0) )
    #plt.show()

    #plot_with_order(expression_trans, row_order, col_order[480:])
    #plot_with_order(samples_choice, row_order, col_order[480:])
    #quit()
    #plot_with_order(expression_trans, row_order, col_order[:100])
    #plot_with_order(samples_choice, row_order, col_order[:100])
    #quit()


    plot_with_order(expression_trans, row_order, col_order, cellTypeIndicator=[cellTypeIndicator])
    plot_with_order(samples_choice, row_order, col_order, cellTypeIndicator=[cellTypeIndicator])
    quit()

    if False:
        for a in range(490, 600):
            unique1, count1 = np.unique(samples_choice[:, col_order[a]] , return_counts=True )
            #unique1, count1 = unique1[count1 > 3], count1[count1 > 3]
            #print (unique1 % 20 / 2, unique1 // 20, count1)
            print (unique1 / 4, count1)
            plt.hist(expression_trans[:, col_order[a]], bins=20)
            plt.show()
    #quit()

    print (type(protein))

    print (protein.shape)
    print (samples_choice.shape)

    #sns.clustermap(expression_trans)
    #plt.show()



    if True:

        corLists1 = []
        corLists2 = []
        corMaxList1 = []
        corMaxList2 = []

        for b in range(protein.shape[1]):
            statList = []

            corLists_mini1 = []
            corLists_mini2 = []
            #for checkIndex in range(2):
            for a in range(samples_choice.shape[1]):
                if np.unique(samples_choice[:, a]).shape[0] >= 2:
                

                    protien_now = protein[:, b]

                    cor = scipy.stats.pearsonr( samples_choice[:, a], protein[:, b] )
                    #cor2 = scipy.stats.pearsonr( expression_trans[:, a], protein[:, b] )
                    cor2 = scipy.stats.pearsonr( scVI_log[:, a], protein[:, b] )
                    
                    #if cor[0] > 0.5:##cor[1] < 0.0001:
                    maxCor = max(abs(cor[0]), abs(cor2[0]))

                    corLists1.append(abs(cor[0]))
                    corLists2.append(abs(cor2[0]))
                    #if cor[0] < 1.1 and cor2[0] < 1.1:
                    corLists_mini1.append(abs(cor[0]))
                    corLists_mini2.append(abs(cor2[0]))

                    if False:#cor[0] < 0.5:
                        unique1 = np.unique(samples_choice[:, a])
                        for c in range(unique1.shape[0]):
                            val1 = unique1[c]
                            stat, pval = ttest_ind(protien_now[samples_choice[:, a] == val1], protien_now[samples_choice[:, a] != val1])
                            if checkIndex == 0:
                                statList.append(stat)

                            else:
                                if stat >= np.max(np.array(statList)) - 1e-8:
                                    if pval < 1e-5:
                                        std1, std2 = np.std(protien_now[samples_choice[:, a] == val1]), np.std(protien_now[samples_choice[:, a] != val1])
                                        stdTotal = ((std1 ** 2) + (std2**2)) ** 0.5
                                        meanDiff = np.mean(protien_now[samples_choice[:, a] == val1]) - np.mean(protien_now[samples_choice[:, a] != val1])
                                        if abs(meanDiff) > 0.5 * stdTotal:
                                            plt.scatter( expression_trans[:, a], protein[:, b] , alpha=0.1 )
                                            #plt.scatter( scVI_log[:, a], protein[:, b]  , alpha=0.1 )
                                            plt.scatter( samples_choice[:, a] / 4 + 0.02, protein[:, b]  , alpha=0.1 )
                                            plt.xlabel('log RNA expression')
                                            plt.ylabel('protein level')
                                            #plt.title('gene: ' + str(a))
                                            plt.title('protein: ' + str(b))
                                            plt.legend(['raw RNA', 'GReinSS'])
                                            #plt.legend(['raw RNA', 'GReinSS'])
                                            plt.show()


                                            #plt.scatter( expression_trans[:, a], protein[:, b] , alpha=0.1 )
                                            plt.scatter( scVI_log[:, a], protein[:, b]  , alpha=0.1 )
                                            plt.scatter( samples_choice[:, a] / 4 + 0.02, protein[:, b]  , alpha=0.1 )
                                            plt.xlabel('log RNA expression')
                                            plt.ylabel('protein level')
                                            #plt.title('gene: ' + str(a))
                                            plt.title('protein: ' + str(b))
                                            plt.legend(['scVI', 'GReinSS'])
                                            #plt.legend(['raw RNA', 'GReinSS'])
                                            plt.show()



                if False:#maxCor > 0.7:# > 0.5:##cor[1] < 0.0001:
                    print ('')
                    print (cor)
                    
                    print (cor2)
            corMax1 = np.max(np.array(corLists_mini1))
            corMax2 = np.max(np.array(corLists_mini2))
            corMaxList1.append(corMax1)
            corMaxList2.append(corMax2)


        #print (np.max(np.array(corMaxList1)))
        #print (np.max(np.array(corLists1)))
        #)
                        
        max1 = np.max(np.array(corLists2))


        plt.scatter(corMaxList2 , corMaxList1)
        plt.plot( [0, max1], [0, max1], color='red' )
        plt.xlabel('scVI expression maximum correlation')
        plt.ylabel('GReinSS expression maxium correlation')
        plt.show()



        
        plt.scatter(corLists2, corLists1)
        plt.plot( [0, max1], [0, max1], color='red' )
        #plt.xlabel('raw expression correlation')
        plt.xlabel('scVI expression correlation')
        plt.ylabel('GReinSS expression correlation')
        plt.show()
    
    quit()




def readAnalaysis():

    data_path = "./data/RNA/raw/"
    protein = pd.read_csv(data_path + "GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv", index_col=0).T
    protein = protein.to_numpy()

    expression = loadnpz('./data/RNA/input/expression.npz')

    scales = loadnpz('./data/RNA/input/scales.npz')
    scales = scales - np.median(scales)
    expression = expression * np.exp2(-scales).reshape((-1, 1))
    expression_trans = np.log2(expression + 1)

    np.random.seed(0)
    expression_trans = expression_trans[np.random.permutation(expression.shape[0])]
    expression_trans[expression_trans>9.5] = 9.5 #For plotting scale

    row_order, col_order = get_clustermap_order(expression_trans)

    arg1 = np.arange(3800 - 600) + 600
    arg2 = np.arange(7800 - 5000) + 5000
    arg1, arg2 = arg1.astype(int), arg2.astype(int)

    print (arg1)

    arg1 = row_order[arg1]
    arg2 = row_order[arg2]

    col_arg = col_order[:120]
    
    

    plot_with_order(expression_trans, row_order, col_order[:120])


    plt.plot(  np.mean( expression_trans[arg1][:, col_arg], axis=0 ) )
    plt.plot(  np.mean( expression_trans[arg2][:, col_arg], axis=0 ) )
    plt.yticks( np.arange(20) / 4, np.arange(20) / 4 )
    plt.show()

    #plot_with_order(samples_choice, row_order, col_order)

    if False:
        for a in range(490, 600):
            unique1, count1 = np.unique(samples_choice[:, col_order[a]] , return_counts=True )
            #unique1, count1 = unique1[count1 > 3], count1[count1 > 3]
            #print (unique1 % 20 / 2, unique1 // 20, count1)
            print (unique1 / 4, count1)
            plt.hist(expression_trans[:, col_order[a]], bins=20)
            plt.show()
    quit()



def saveExpression():


    data_path = "./data/RNA/raw/"
    #preprocessing
    expression = pd.read_csv(data_path + "GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv", index_col=0).T
    protein = pd.read_csv(data_path + "GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv", index_col=0).T
    norm_protein = pd.read_csv(data_path + "GSE100866_CBMC_8K_13AB_10X-ADT_clr-transformed.csv", index_col=0).T
    gene_names = expression.columns
    barcodes = expression.index
    protein_names = protein.columns
    #print expression.shape[0], " cells with ", expression.shape[1], " genes and ", norm_protein.shape[1], "proteins"
    #8617  cells with  36280  genes and  13 proteins

    

    selected = np.std(expression.values, axis=0).argsort()[-600:][::-1]
    selected = np.sort(selected)
    expression = expression.values[:, selected]


    np.savez_compressed('./data/RNA/input/expression.npz', expression)




def saveExpScales():


    data_path = "./data/RNA/raw/"
    #preprocessing
    #expression = pd.read_csv(data_path + "GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv", index_col=0).T

    expression = loadnpz('./data/RNA/input/expression_train.npz')
    norm_protein = pd.read_csv(data_path + "GSE100866_CBMC_8K_13AB_10X-ADT_clr-transformed.csv", index_col=0).T
    #gene_names = expression.columns
    #barcodes = expression.index
    #expression = expression.values


    #print (np.mean(expression))
    #print (np.max(np.mean(expression, axis=0)))
    #quit()


    expression_transform = np.log2(expression + 1)
    meanGene = np.mean(expression_transform, axis=0)
    
    argHigh = np.argwhere(meanGene > 2)[:, 0]
    #stdValues = np.std(expression_transform[:, argHigh], axis=0)

    scales = np.mean(expression_transform[:, argHigh], axis=1)
    #medianScale = np.median(scale)

    np.savez_compressed('./data/RNA/input/scales_train.npz', scales)


#saveExpScales()
#quit()


def commandList():

    #aws s3 cp --no-sign-request \
    #s3://openproblems-data/resources/datasets/openproblems_v1/pancreas/l1_sqrt/dataset.h5ad \
    #pancreas.dataset.h5ad

    True 

#readAnalaysis()
#quit()


def loadPancreas():

    import anndata as ad

    adata = ad.read_h5ad("./data/RNA/raw/pancreas.dataset.h5ad")

    X_counts = adata.layers["counts"]
    X_counts = X_counts.toarray()

    X_counts = X_counts[np.random.permutation(X_counts.shape[0])][:, np.random.permutation(X_counts.shape[1])]

    #total_counts = np.mean(X_counts, axis=0)

    sns.clustermap(np.log(X_counts[:1000, :1000]+1))
    plt.show()
    #quit()

    total_counts = np.mean(X_counts, axis=0)
    total_counts = np.sort(total_counts)
    print (np.sum(  total_counts[-1000:])  / np.sum(total_counts) )

    print (total_counts.shape)

    percentage = np.cumsum(total_counts)
    percentage = percentage / percentage[-1]

    plt.plot(percentage)
    plt.show()

    #print (X_counts.shape)

    #sns.clustermap(np.log2(X_counts[:500, :]+1))
    #plt.show()


#loadPancreas()
#quit()


def mcv_split(X, p=0.5):
    # X = raw counts (cells x genes)

    train = np.random.binomial(X, p)
    test = X - train

    return train, test


def saveTrainExpression():

    expression = loadnpz('./data/RNA/input/expression.npz')
    expression_train, expression_test = mcv_split(expression, p=0.5)

    np.savez_compressed('./data/RNA/input/expression_train.npz', expression_train)
    np.savez_compressed('./data/RNA/input/expression_test.npz', expression_test)

    


def runVI():

    import scvi

    expression = loadnpz('./data/RNA/input/expression_train.npz')

    import anndata as ad

    adata = ad.AnnData(expression)
    

    scvi.model.SCVI.setup_anndata(adata)

    model = scvi.model.SCVI(adata)
    #model.train(max_epochs=100)
    model.train(max_epochs=1000)

    # denoised expression
    #denoised = model.get_normalized_expression()
    denoised = model.get_normalized_expression(library_size="latent")

    np.savez_compressed('./data/RNA/scVI/denoised2.npz', denoised)

#runVI()
#quit()


def giveMSE(pred, target):

    return np.mean((pred - target)**2)

def poisson_loss(pred, target):
    return np.mean(pred - target * np.log(pred + 1e-10))

def calculateError():

    pred = loadnpz('./data/RNA/pred/100Pred_train_noNoise.npz').astype(float)
    #pred = loadnpz('./data/RNA/pred/100Pred_train.npz').astype(float)
    
    
    
    expression_train = loadnpz('./data/RNA/input/expression_train.npz')
    np.random.seed(0)
    perm1 = np.random.permutation(expression_train.shape[0])
    perm1 = perm1[:pred.shape[0]]

    expression_train = expression_train[perm1]
    expression_test = loadnpz('./data/RNA/input/expression_test.npz')[perm1]
    denoised = loadnpz('./data/RNA/scVI/denoised2.npz')[perm1]

    scales = loadnpz('./data/RNA/input/scales_train.npz')[perm1]
    scales_adj = scales - np.median(scales)
    scales_adj = scales_adj.reshape((scales_adj.shape[0], 1))

    
    pred = np.exp2((pred * 0.25) + scales_adj) #- 0.9
    #pred = np.exp2((pred * 0.25) ) - 0.9


    pred = pred / np.mean(pred + 0.1, axis=1).reshape((-1, 1))
    pred = pred * np.mean(expression_train + 0.1, axis=1).reshape((-1, 1))
    #pred = pred / np.mean(pred + 1, axis=0).reshape((1, -1))
    #pred = pred * np.mean(expression_train+ 1, axis=0).reshape((1, -1))

    #sns.clustermap(np.log2(pred+1))
    #plt.show()

    #print (np.mean(pred))
    #print (np.mean(expression_train))
    #print (np.mean(expression_test))

    #print (scipy.stats.pearsonr( denoised.reshape((-1,)), expression_test.reshape((-1,)) ))
    #print (scipy.stats.pearsonr( expression_train.reshape((-1,)), expression_test.reshape((-1,)) ))
    #print (scipy.stats.pearsonr( pred.reshape((-1,)), expression_test.reshape((-1,)) ))

    #print ('')
    #print (scipy.stats.pearsonr( denoised[:, 0], expression_test[:, 0] ))
    #print (scipy.stats.pearsonr( expression_train[:, 0], expression_test[:, 0] ))
    #print (scipy.stats.pearsonr( pred[:, 0], expression_test[:, 0] ))
    #print (scipy.stats.pearsonr( pred[:, 0], expression_train[:, 0] ))

    #plt.scatter(pred.reshape((-1,)), expression_test.reshape((-1,)) )
    #plt.scatter(pred[:, 1], expression_train[:, 1])
    #plt.show()
    #quit()
    lam = 0.1
    middle = (pred * lam) + (expression_train * (1-lam))   
    middle2 = (denoised * lam) + (expression_train * (1-lam))   
    print (poisson_loss(denoised, expression_test))
    print (poisson_loss(expression_train, expression_test))
    print (poisson_loss(middle, expression_test))
    print (poisson_loss(middle2, expression_test))
    #print (poisson_loss(pred, expression_test))
    print ('')
    

    #sns.clustermap( np.log2(denoised+1) )
    #plt.show()

    quit()


#calculateError()
#quit()


def getPBMC():

    import scvi
    from scvi.data import pbmc_seurat_v4_cite_seq


    #adata = scvi.data.pbmcs_10x_cite_seq()
    adata = pbmc_seurat_v4_cite_seq()
    #adata = scvi.data.pbmcs_10x_cite_seq()


    print(adata.obsm.keys())
    print(adata.obs.columns)
    quit()


    if False:
        X_counts = adata.X
        X_counts = X_counts.toarray()

        np.savez_compressed('./data/RNA/input/PBMC.npz', X_counts)
        print (X_counts.shape)

    geneNames = adata.var_names.to_numpy()
    print (geneNames)
    np.savez_compressed('./data/RNA/input/PBMC_geneNames.npz', geneNames)
    quit()



    cellType2 = adata.obs["celltype.l2"].to_numpy()
    cellType1 = adata.obs["celltype.l1"].to_numpy()
    #protein = adata.obsm["protein_expression"]

    print (cellType1)
    

    np.savez_compressed('./data/RNA/input/PBMC_cellType1.npz', cellType1)
    np.savez_compressed('./data/RNA/input/PBMC_cellType2.npz', cellType2)
    #np.savez_compressed('./data/RNA/input/PBMC_protein.npz', protein)
    True



#getPBMC()
#quit()

def processPBMC():

    cellType1 = loadnpz('./data/RNA/input/PBMC_cellType1.npz', allow_pickle=True)
    cellType2 = loadnpz('./data/RNA/input/PBMC_cellType2.npz', allow_pickle=True)

    #print (np.unique(cellType1, return_counts=True))
    #print (np.unique(cellType2_subset, return_counts=True))
    #quit()


    cellTypeGood = np.array(['CD4 T', 'CD8 T', 'other T' ])
    argTCell =  np.argwhere(  np.isin( cellType1, cellTypeGood ) )[:, 0]


    cellType1_subset = cellType1[argTCell]
    cellType2_subset = cellType2[argTCell]

    #print (argTCell.shape)
    #print (np.unique( cellType1_subset, return_counts=True ))
    #print (np.unique( cellType2_subset, return_counts=True ))
    #quit()

    

    X_counts = loadnpz('./data/RNA/input/PBMC.npz')
    X_counts = X_counts[argTCell]



    #geneSum = np.sum(X_counts, axis=0)
    geneSum = np.std(X_counts, axis=0)
    topGenes = np.argsort(geneSum * -1)

    #X_counts = X_counts[:, topGenes[:500]]
    #X_counts = X_counts[:, topGenes[:1000]]

    print (np.unique(cellType1_subset, return_counts=True))
    print (np.unique(cellType2_subset, return_counts=True))


    #print (X_counts.shape)
    #quit()

    #X_counts_mini = X_counts[np.random.permutation(X_counts.shape[0])[:1000]]

    
    #X_counts_mini = X_counts_mini[:, np.argsort(geneSum)]

    #geneSum_sort = np.sort(geneSum)
    #geneSum_sort = np.cumsum(geneSum_sort)
    #geneSum_sort = geneSum_sort / geneSum_sort[-1]

    #print (geneSum_sort[-500])



    #numInclude = np.argwhere( geneSum_sort[-1::-1] > 0.7 )[:, 0][-1]
    #print (numInclude)
    #quit()
    #plt.plot(geneSum_sort)
    #plt.show()
    
    #[:, np.random.permutation(X_counts.shape[1])[:1000]]
    
    #print (np.unique(cellType1))

    #sns.heatmap(  np.log2(X_counts_mini + 1) )

    #sns.clustermap(  np.log2(X_counts[ np.random.permutation(X_counts.shape[0])[:500] , :] + 1) )
    #plt.show()

    #np.savez_compressed('./data/RNA/input/PBMC_Tcell_top1000.npz', X_counts)
    #np.savez_compressed('./data/RNA/input/PBMC_Tcell_top1000_cellType1.npz', cellType1_subset)
    #np.savez_compressed('./data/RNA/input/PBMC_Tcell_top1000_cellType2.npz', cellType2_subset)

    np.savez_compressed('./data/RNA/input/PBMC_Tcell.npz', X_counts)
    np.savez_compressed('./data/RNA/input/PBMC_Tcell_cellType1.npz', cellType1_subset)
    np.savez_compressed('./data/RNA/input/PBMC_Tcell_cellType2.npz', cellType2_subset)


#processPBMC()
#quit()




def trainLOPG(expression, meanFactor, scales, loadModels=[]):


    #device = torch.device("cpu")
    device = torch.device("mps")

    torch.manual_seed(0)

    Ngenes = expression.shape[1]
    NHidden = 1000

    numMean = meanFactor.shape[0]

    specialSampler = True


    if len(loadModels) == 0:
        NHiddenReg = 40
        Ngroup = 5
        #NHiddenReg = 5
        #NHiddenReg = 10
        if specialSampler:
            encodeModel = SpecialEncoder(Ngenes, NHidden, Ngroup, numMean).to(device)
        else:
            encodeModel = Encoder(Ngenes, NHidden, numMean).to(device)
        autoregModel = quickPolicy(Ngenes, numMean, NHiddenReg).to(device)
        #NHiddenReg = 5
        #autoregModel = AutoregressiveMatrixModel(Ngenes, numMean, NHiddenReg).to(device)
    else:
        encodeModel = torch.load(loadModels[0]).to(device)
        autoregModel = torch.load(loadModels[1]).to(device)


    #if True:
    #    autoregModel = torch.load('./data/RNA/model/onPol_PBMC_2000Genes_noCross_2.pt').to(device)
    
    
    expression = expression.to(device)
    scales = scales.to(device)
    meanFactor = meanFactor.to(device)

    #batchSize = 1000
    batchSize = 100
    #batchSize = 50
    #batchSize = 250
    #batchSize = 10
    #batchSize = 8000
    Nbatch = expression.shape[0] // batchSize

    #crossSize = 100
    #crossSize = 250 #Good
    #crossSize = 500
    #crossSize = 1000

    #crossSize = min(crossSize, batchSize)

    Niter = 10000
    #Niter = 100000

    #meanFactor = torch.zeros((expression.shape[0], expression.shape[1], numMean ))
    #meanFactor[:, :, torch.arange(numMean)] =  torch.arange(numMean) * stepSize
    
    
    



    #lr_on = 1e-3 
    lr_off = 1e-4 

    #lr_on = 1e-4 #Currently used for full dataset
    #lr_off = 1e-5 #Currently used for full dataset

    lr_on = 1e-10

    #lr_on = 1e-5
    #lr_off = 1e-5
    #lr_off = 1e-7

    noNoise = True
    #noNoise = False

    #rewardType = 1
    #rewardType = 2
    rewardType = 0

    #dupGen = 5
    dupGen = 2
    #dupGen = 1

    print ('rewardType', rewardType)
    #print ("NHidden", NHiddenReg)
    print ('noNoise', noNoise)
    print ('datasize', expression.shape)
    print ('dupGen', dupGen)


    cumProb = False
    if rewardType >= 1:
        cumProb = True
    #cumProb = False 


    
    

    #optimizer_onPolicy = torch.optim.RMSprop(autoregModel.parameters(), lr=lr_on , alpha=0.9)
    #optimizer_offPolicy = torch.optim.RMSprop(encodeModel.parameters(), lr=lr_off , alpha=0.9)
    

    optimizer_onPolicy = torch.optim.RMSprop(autoregModel.parameters(), lr=lr_on , alpha=0.998)
    optimizer_offPolicy = torch.optim.RMSprop(encodeModel.parameters(), lr=lr_off , alpha=0.998)
    
    
    for iter in range(Niter):

        probX_list = []

        probX_paste = np.zeros(  Nbatch * batchSize )

        #perm2 = np.random.permutation(expression.shape[0])
        time2 = time.time()
        for batchIndex in range(Nbatch):

            #print ('batchIndex', batchIndex, Nbatch)

            timeList = []
            #torch.mps.synchronize()
            timeList.append(time.time())

            argBatch = np.arange(batchSize) + (batchIndex * batchSize)


            batchNow = expression[argBatch]
            scales_batch = scales[argBatch]


            timeList.append(time.time()) #1

            stdFactor = autoregModel.stdFactor[:, :, 0]
            allExpressionProbs_batch = doGaussian(batchNow, meanFactor, stdFactor, scales_batch, noNoise=noNoise)
            

            timeList.append(time.time()) #2

            groupProbs, offPolicyProb_all = encodeModel(torch.log2(batchNow+1))
            arange1 = torch.arange(batchSize * dupGen)  #// batchSize
            groupProbs = groupProbs.repeat((dupGen, 1))
            offPolicyProb_all = offPolicyProb_all.repeat((dupGen, 1, 1, 1))
            allExpressionProbs_batch = allExpressionProbs_batch.repeat((dupGen, 1, 1))

            #print (torch.exp(groupProbs))
            #quit()
            

            selectedGroup = gumbel_sample(groupProbs)
            offPolicyProb_dup = offPolicyProb_all[arange1, selectedGroup]
            

            
            
            timeList.append(time.time()) #3

            if True:
                offPolicyProb_dup = offPolicyProb_dup + allExpressionProbs_batch.detach()
            offPolicyProb_dup = torch.log_softmax(offPolicyProb_dup, axis=2)  

            
            timeList.append(time.time()) #4

            #offPolicyProb_dup = offPolicyProb.repeat(dupGen, 1, 1)
            #offPolicyProb_dup = offPolicyProb_dup.reshape(( dupGen, batchSize, offPolicyProb_dup.shape[1], offPolicyProb_dup.shape[2] ))

            samples = gumbel_sample(offPolicyProb_dup)
            samples_oneHot = F.one_hot(samples, num_classes=numMean).float()
            
            #torch.mps.synchronize()
            timeList.append(time.time()) #5


            onPolicyProb = autoregModel(samples)
            
            entropy = torch.mean(torch.sum(torch.exp(onPolicyProb) * onPolicyProb, axis=(1, 2))) * -1
            #offEntropy = torch.mean(torch.sum(torch.exp(offPolicyProb) * offPolicyProb, axis=(1, 2))) * -1

            #torch.mps.synchronize()
            timeList.append(time.time()) #6


            offPolicyProb_samples = getProbGroupSamples_onehot(offPolicyProb_all,  groupProbs, samples_oneHot, cumProb=cumProb)
            onPolicyProb_samples = getProbSamples_onehot(onPolicyProb, samples_oneHot, cumProb=cumProb)
            
            timeList.append(time.time()) #8


            #print (allExpressionProbs_batch.shape, samples_oneHot.shape, offPolicyProb_samples.shape, onPolicyProb_samples.shape)
            #quit()
            
            probX, rewards = getObservationProbs(allExpressionProbs_batch, samples_oneHot, offPolicyProb_samples, onPolicyProb_samples, dupGen)


            rewards_print = rewards.cpu().data.numpy()
            rewards_print = rewards_print.reshape((2, rewards_print.shape[0] // 2)) 
            rewards_print = np.sort(rewards_print, axis=0)
            #print (rewards_print.T)
            print ('mean1', np.mean(rewards_print[0]))
            
            

            probX_list.append(torch.mean(probX).cpu().data.numpy())

            probX_paste[argBatch] = probX.cpu().data.numpy()

            #torch.mps.synchronize()
            timeList.append(time.time()) #9

            #print (onPolicyProb_samples.shape, offPolicyProb_samples.shape)
            #print (rewards_summed.shape, rewards.shape)
            #quit()


            
            loss_onPolicy = -1 * torch.mean(rewards.detach() * onPolicyProb_samples)
            #if not noNoise:
            loss_onPolicy = loss_onPolicy - torch.mean(probX) #Specifically for the autoregModel.stdFactor
            
            loss_offPolicy = -1 * torch.mean(rewards.detach() * offPolicyProb_samples)
            
            
            #torch.mps.synchronize()
            timeList.append(time.time())

            optimizer_onPolicy.zero_grad()
            loss_onPolicy.backward()
            optimizer_onPolicy.step()

            optimizer_offPolicy.zero_grad()
            loss_offPolicy.backward()
            #torch.nn.utils.clip_grad_norm_(encodeModel.parameters(), max_norm=1e-4)
            optimizer_offPolicy.step()

            #print ("F")

            #torch.mps.synchronize()
            timeList.append(time.time())


            timeList = np.array(timeList)
            timeList = timeList[1:] - timeList[:-1]
            #
            #print (np.round(timeList, decimals=2))
            #print (np.sum(timeList))
            #quit()


        #print ('inverse1', np.unique(inverse1).shape)
        print (time.time() - time2)
        #quit()


        probX_list = np.array(probX_list)

        #if iter % 20 == 0:
        if iter % 1 == 0:
            #for printIndex in range(10):
            #    print ('')
            print ("Iter", iter)
            print ('probX', np.mean(probX_list))
            #print ('probX-2', np.mean(probX_paste))

            #probX_paste_sort = np.sort(probX_paste)
            #SizeChunk = probX_paste_sort.shape[0] // 4
            #print ('sorted: ', probX_paste_sort[0], probX_paste_sort[SizeChunk], probX_paste_sort[2*SizeChunk], probX_paste_sort[3*SizeChunk], probX_paste_sort[-1])




            print ('entropy', entropy)
            #print ('offEntropy', offEntropy)
            #quit()

            torch.save(encodeModel,  './data/RNA/model/offPol_PBMC_2000Genes_noCross_4.pt') 
            torch.save(autoregModel,  './data/RNA/model/onPol_PBMC_2000Genes_noCross_4.pt')



def sampleToOneHot(samples_now, numDist):


    argAll = np.argwhere(samples_now > -1)
    oneHot = torch.zeros((samples_now.shape[0], samples_now.shape[1], numDist))
    oneHot[argAll[:, 0], argAll[:, 1], samples_now[argAll[:, 0], argAll[:, 1]] ] = 1

    return oneHot



def analyzePredCells():


    #prediction = loadnpz('./data/RNA/pred/pred100_PBMC_top2000_simple_noNoise.npz')
    prediction = loadnpz('./data/RNA/pred/pred200_PBMC_top2000_simple_noNoise.npz')

    geneNames = loadnpz('./data/RNA/input/PBMC_geneNames.npz', allow_pickle=True)
    expression = loadnpz('./data/RNA/input/PBMC_Tcell.npz', allow_pickle=True)

    

    cellType1 = loadnpz('./data/RNA/input/PBMC_Tcell_cellType1.npz', allow_pickle=True)

    np.random.seed(0)
    perm1 = np.random.permutation(expression.shape[0])
    perm1 = perm1[:prediction.shape[0]]

    scales = loadnpz('./data/RNA/input/PBMC_Tcell_scales.npz')
    scales = scales - np.median(scales)
    scales = scales[perm1]


    #cellType2 = loadnpz('./data/RNA/input/PBMC_Tcell_cellType2.npz', allow_pickle=True)
    col_order = np.argsort( np.std(expression, axis=0) * -1 )
    col_order = col_order[:2000]
    expression = expression[:, col_order]
    expression = expression[perm1]
    cellType1 = cellType1[perm1]

    geneNames = geneNames[col_order]

    

    argGene1 = np.argwhere(geneNames == 'CD4')[0, 0]
    argGene2 = np.argwhere(geneNames == 'CD8A')[0, 0]
    argGene3 = np.argwhere(geneNames == 'CD8B')[0, 0]

    #print (argGene1)
    #quit()
    print (argGene1)
    exp1 = expression[:, argGene1]
    exp2 = expression[:, argGene2] + expression[:, argGene3]
    #expression = expression[:, argGene]


    #plt.hist(scales, bins=100)
    #plt.show()


    #plt.hist(np.exp2(scales) * exp1, bins=100)
    #plt.show()

    #quit()

    print ('0', np.argwhere(np.logical_and(exp1[cellType1 == 'CD4 T'] == 0,  exp2[cellType1 == 'CD4 T']==0) ).shape)
    print ('CD4 bad', np.argwhere(np.logical_and(exp1[cellType1 == 'CD4 T'] == 0,  exp2[cellType1 == 'CD4 T']>=1) ).shape)
    print ('CD4', np.argwhere(np.logical_and(exp1[cellType1 == 'CD4 T'] >= 1,  exp2[cellType1 == 'CD4 T']==0) ).shape)
    print ('CD8 0', np.argwhere(np.logical_and(exp1[cellType1 == 'CD8 T'] == 0,  exp2[cellType1 == 'CD8 T']==0) ).shape)
    print ('CD8 good',np.argwhere(np.logical_and(exp1[cellType1 == 'CD8 T'] == 0,  exp2[cellType1 == 'CD8 T']>=1) ).shape)
    print ('CD8 bad',np.argwhere(np.logical_and(exp1[cellType1 == 'CD8 T'] >= 1,  exp2[cellType1 == 'CD8 T']==0) ).shape)

    

    #print (np.mean(exp1))
    #print (np.mean(exp2))

    #sns.clustermap( np.log2(expression + 1)[:, :]  )
    #plt.show()

    #prediction[prediction > 4] = 4 
    #sns.heatmap(prediction[:, :]  )
    #plt.show()


    #plt.hist(exp1, bins=100)
    #plt.show()
    #quit()

    plt.scatter( exp1[cellType1 == 'CD4 T'], exp2[cellType1 == 'CD4 T'] ,alpha=0.05 )
    plt.scatter( exp1[cellType1 == 'CD8 T'], exp2[cellType1 == 'CD8 T'] ,alpha=0.05 )
    plt.show()

    exp1_copy = np.copy(exp1)
    exp2_copy = np.copy(exp2)

    exp1 = prediction[:, argGene1]
    exp2 = prediction[:, argGene2] + prediction[:, argGene3]

    print ("new - old")
    plt.scatter(exp1_copy, exp1, alpha=0.5)
    plt.show()

    plt.scatter(exp2_copy, exp2, alpha=0.5)
    plt.show()

    print ('')
    print ('0', np.argwhere(np.logical_and(exp1[cellType1 == 'CD4 T'] == np.min(exp1),  exp2[cellType1 == 'CD4 T']== np.min(exp2) ) ).shape)
    print ('CD4 bad', np.argwhere(np.logical_and(exp1[cellType1 == 'CD4 T'] == np.min(exp1),  exp2[cellType1 == 'CD4 T'] > np.min(exp2) ) ).shape)
    print ('CD4', np.argwhere(np.logical_and(exp1[cellType1 == 'CD4 T'] > np.min(exp1),  exp2[cellType1 == 'CD4 T']== np.min(exp2) ) ).shape)
    print ('CD8 0', np.argwhere(np.logical_and(exp1[cellType1 == 'CD8 T'] == np.min(exp1),  exp2[cellType1 == 'CD8 T']== np.min(exp2) ) ).shape)
    print ('CD8 good',np.argwhere(np.logical_and(exp1[cellType1 == 'CD8 T'] == np.min(exp1),  exp2[cellType1 == 'CD8 T'] > np.min(exp2) ) ).shape)
    print ('CD8 bad',np.argwhere(np.logical_and(exp1[cellType1 == 'CD8 T'] > np.min(exp1),  exp2[cellType1 == 'CD8 T']== np.min(exp2) ) ).shape)
    #quit()

    #print (np.mean(exp1))
    #print (np.mean(exp2))
    #quit()

    plt.scatter( exp1[cellType1 == 'CD4 T'], exp2[cellType1 == 'CD4 T'] ,alpha=0.05 )
    plt.scatter( exp1[cellType1 == 'CD8 T'], exp2[cellType1 == 'CD8 T'] ,alpha=0.05 )
    plt.show()

    quit()







#expression = loadnpz('./data/RNA/input/PBMC_Tcell_top500.npz', allow_pickle=True)
#expression = loadnpz('./data/RNA/input/PBMC_Tcell_top1000.npz', allow_pickle=True)
expression = loadnpz('./data/RNA/input/PBMC_Tcell.npz', allow_pickle=True)
col_order = np.argsort( np.std(expression, axis=0) * -1 )


#print (np.argwhere( col_order ==  9723 ))
#print (np.argwhere( col_order ==  2089 ))
#print (np.argwhere( col_order ==  2090 ))
#quit()

#expression = expression[:, col_order[:50]]
#expression = expression[:, col_order[:100]]
expression = expression[:, col_order[:500]]
#expression = expression[:, col_order[:2000]]
#expression = expression[:, col_order]


scales = loadnpz('./data/RNA/input/PBMC_Tcell_scales.npz')
scales = scales - np.median(scales)


numMean = 40
stepSize = 0.4
meanFactor = (torch.arange(numMean) - 12) * stepSize


np.random.seed(0)
perm1 = np.random.permutation(expression.shape[0])
expression = expression[perm1]


expression = torch.tensor(expression).float()
scales = scales[perm1]
scales = torch.tensor(scales).float()


offModelFile = './data/RNA/model/offPol_PBMC_2000Genes_noCross_3.pt'
onModelFile = './data/RNA/model/onPol_PBMC_2000Genes_noCross_3.pt'




trainLOPG(expression, meanFactor, scales)#, loadModels=[offModelFile, onModelFile])
quit()









