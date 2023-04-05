import numpy as np
from .DiscreteD import DiscreteD
from .GaussD import GaussD
from .MarkovChain import MarkovChain


class HMM:
    """
    HMM - class for Hidden Markov Models, representing
    statistical properties of random sequences.
    Each sample in the sequence is a scalar or vector, with fixed DataSize.
    
    Several HMM objects may be collected in a single multidimensional array.
    
    A HMM represents a random sequence(X1,X2,....Xt,...),
    where each element Xt can be a scalar or column vector.
    The statistical dependence along the (time) sequence is described
    entirely by a discrete Markov chain.
    
    A HMM consists of two sub-objects:
    1: a State Sequence Generator of type MarkovChain
    2: an array of output probability distributions, one for each state
    
    All states must have the same class of output distribution,
    such as GaussD, GaussMixD, or DiscreteD, etc.,
    and the set of distributions is represented by an object array of that class,
    although this is NOT required by general HMM theory.
    
    All output distributions must have identical DataSize property values.
    
    Any HMM output sequence X(t) is determined by a hidden state sequence S(t)
    generated by an internal Markov chain.
    
    The array of output probability distributions, with one element for each state,
    determines the conditional probability (density) P[X(t) | S(t)].
    Given S(t), each X(t) is independent of all other X(:).
    
    
    References:
    Leijon, A. (20xx) Pattern Recognition. KTH, Stockholm.
    Rabiner, L. R. (1989) A tutorial on hidden Markov models
    	and selected applications in speech recognition.
    	Proc IEEE 77, 257-286.
    
    """
    def __init__(self, mc, distributions):

        self.stateGen = mc  # Markov chain with defined q and A
        self.outputDistr = distributions  # All possible distributions of output sources

        self.nStates = mc.nStates  # number of possible states
        self.dataSize = distributions[0].dataSize  # dimension of random variable Xt
    
    def rand(self, nSamples):
        """
        [X,S]=rand(self,nSamples); generates a random sequence of data
        from a given Hidden Markov Model.
        
        Input:
        nSamples=  maximum no of output samples (scalars or column vectors)
        
        Result:
        X= matrix or row vector with output data samples
        S= row vector with corresponding integer state values
          obtained from the self.StateGen component.
          nS= length(S) == size(X,2)= number of output samples.
          If the StateGen can generate infinite-duration sequences,
              nS == nSamples
          If the StateGen is a finite-duration MarkovChain,
              nS <= nSamples
        """
        
        #*** Insert your own code here and remove the following error message 
        
        # Generate the state sequence
        S = self.stateGen.rand(nSamples)
        
        # Initialize the output sequence X
        nS = S.size
        X = np.zeros((self.dataSize, nS))
        
        # Generate X according to the distributions determined by states
        for i in range(nS):
            X[:,i] = self.outputDistr[S[i]-1].rand(1).reshape(-1)
        
        return X, S
            
                                
    def viterbi(self):
        pass

    def train(self):
        pass

    def stateEntropyRate(self):
        pass

    def setStationary(self):
        pass
    
    def Get_px(self, x):
        """
        Calculate the The sclaed state-conditional probability mass or density value matrix px
        px = [f(x1|s1) f(x2|s1) ... f(xt|s1]
             [f(x1|s2) f(x2|s2) ... f(xt|s2]
             [f(x1|sn) f(x2|sn) ... f(xt|sn]
        and the scale factors scaler_px
        """
        # Initialization 
        prob = np.zeros((self.nStates, len(x)))
        px = np.zeros((self.nStates, len(x)))
        scaler_px = np.zeros(len(x), )
        
        for j in range(len(x)):
            for i in range(self.nStates):
                prob[i, j] = self.outputDistr[i].prob(x[j])
            scaler_px[j] = np.max(prob[:, j])
            
        for j in range(len(x)):
            for i in range(self.nStates):
                px[i, j] = prob[i, j] / scaler_px[j]
            
        
        return px, scaler_px
    

    def logprob(self, x):
        """
          Calculate the log P(X=x|lambda)
        
        """
        # Generate px and scaler factors
        px, scaler_px = self.Get_px(x)
        
        # Generate alfahat and scaled c
        alfahat, c = self.stateGen.forward(px)
        
        # c_r is the original, unscaled version of c
        c_r = np.zeros(len(c), )
        c_r = c
        
        for i in range(len(scaler_px)):
            c_r[i] = c[i] * scaler_px[i]
            
        logP = np.sum(np.log(c_r))
        
        return logP

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass