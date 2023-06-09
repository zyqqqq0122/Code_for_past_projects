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
            X[:, i] = self.outputDistr[S[i]-1].rand(1).reshape(-1)
        
        return X, S
            
                                
    def viterbi(self):
        pass

    def train(self, x):
        """
        Training the HMM with Baum-Welch algorithm.

        Input:
        x= the matrix of observed sequence
        x.shape[0] = number of clips for training
        x.shape[1] = 13 = self.dataSize = number of MFCCs
        x.shape[2] = number of frames in each clip

        Result:
        A_new = the updated transition matrix A
        cov_new= the updated covariance matrix C

        """
        # Calculation of alfahat, betahat and gamma
        # prob, px, scalar = self.Get_px(x)
        # alfahat, c = self.stateGen.forward(px)
        # betahat = self.stateGen.backward(px, c)
        # c_t = np.zeros((self.nStates, alfahat.shape[1]))
        # for i in range(self.nStates):
        #     c_t[i, :] = c[:alfahat.shape[1]] * scalar
        # gamma = alfahat * betahat * c_t

        # Updating cov_new with EM algorithm
        # nomi = np.zeros((x.shape[0], x.shape[0]))
        # nomi = 0
        # denomi = 0
        # var_new = [None] * self.nStates
        # for i in range(self.nStates):
        #     for t in range(x.shape[1]):
        #         denomi += gamma[i, t]
        #         # nomi += gamma[i, t] * np.dot(x[:, t].reshape(-1, 1), x[:, t].reshape(1, -1))
        #         nomi += gamma[i, t] / x.shape[0] * np.dot(x[:, t].reshape(1, -1), x[:, t].reshape(-1, 1))
        #     var_new[i] = nomi / denomi

        # Updating A
        # A = self.stateGen.A
        # A_new = np.zeros((A.shape[0], A.shape[1]))
        # xibar = np.zeros((A.shape[0], A.shape[1]))
        # for i in range(xibar.shape[0]):
        #     for j in range(xibar.shape[1]-1):
        #         for t in range(x.shape[1]-1):
        #             xibar[i, j] += alfahat[i, t]*A[i, j]*prob[j, t+1]*betahat[j, t+1]
        # xibar[-1, -1] = gamma[-1, -1]
        #
        # for i in range(A_new.shape[0]):
        #     for j in range(A_new.shape[1]):
        #         A_new[i, j] = xibar[i, j] / np.sum(xibar[i, :])

        # return var_new, A_new, c_t, alfahat, betahat


        # Calculation of alfahat, betahat and gamma of each clip r
        alfahat = np.zeros((len(x), self.nStates, x[0].shape[1]))
        betahat = np.zeros((len(x), self.nStates, x[0].shape[1]))
        prob = np.zeros((len(x), self.nStates, x[0].shape[1]))
        gamma = np.zeros((len(x), self.nStates, x[0].shape[1]))
        for r in range(len(x)):
            prob[r, :, :], px, scalar = self.Get_px(x[r])
            alfahat[r, :, :], c = self.stateGen.forward(px)
            betahat[r, :, :] = self.stateGen.backward(px, c)
            c_t = np.zeros((len(x), self.nStates, alfahat[r].shape[1]))
            for i in range(self.nStates):
                c_t[r, i, :] = c[:alfahat[r].shape[1]] * scalar
            gamma[r, :, :] = alfahat[r, :, :] * betahat[r, :, :] * c_t[r, :, :]


        # Updating A
        A = self.stateGen.A
        A_new = np.zeros((A.shape[0], A.shape[1]))
        xibar = np.zeros((len(x), A.shape[0], A.shape[1]))
        for r in range(len(x)):
            for i in range(xibar.shape[1]):
                for j in range(xibar.shape[2] - 1):
                    for t in range(x[0].shape[1] - 1):
                        xibar[r, i, j] += alfahat[r, i, t] * A[i, j] * prob[r, j, t + 1] * betahat[r, j, t + 1]
            xibar[r, -1, -1] = gamma[r, -1, -1]

        for i in range(A_new.shape[0]):
            for j in range(A_new.shape[1]):
                A_new[i, j] = np.sum(xibar[:, i, j]) / np.sum(xibar[:, i, :])

        # Updating covariance matrix
        nomi = np.zeros((x[0].shape[0], x[0].shape[0]))
        # nomi = 0
        denomi = 0
        cov_new = [None] * self.nStates

        for i in range(self.nStates):
            denomi = np.sum(gamma[:, i, :])
            for r in range(len(x)):
                for t in range(x[0].shape[1]):
                    nomi += gamma[r, i, t] * np.dot(x[r][:, t].reshape(-1, 1), x[r][:, t].reshape(1, -1))
            cov_new[i] = nomi / denomi

        return A_new, cov_new

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
        prob = np.zeros((self.nStates, x.shape[1]))
        px = np.zeros((self.nStates, x.shape[1]))
        scaler_px = np.zeros(x.shape[1], )
        
        for j in range(x.shape[1]):
            for i in range(self.nStates):
                prob[i, j] = self.outputDistr[i].prob(x[:, j])
            scaler_px[j] = np.max(prob[:, j])
            
        for j in range(x.shape[1]):
            for i in range(self.nStates):
                px[i, j] = prob[i, j] / scaler_px[j]
            
        
        return prob, px, scaler_px
    

    def logprob(self, x):
        """
          Calculate the log P(X=x|lambda)
        
        """
        # Generate px and scaler factors
        prob, px, scaler_px = self.Get_px(x)
        
        # Generate alfahat and scaled c
        alfahat, c = self.stateGen.forward(px)
        
        # c_r is the original, unscaled version of c
        c_r = np.zeros(len(c), )
        c_r = c
        
        for i in range(len(scaler_px)):
            c_r[i] = c[i] * scaler_px[i]
            # if c_r[i] == 0:
            #     c_r[i] = 0.00000001
            
        logP = np.sum(np.log(c_r))
        
        return logP

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass