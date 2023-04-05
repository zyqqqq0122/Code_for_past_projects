import numpy as np
from .DiscreteD import DiscreteD

class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]


        self.nStates = transition_prob.shape[0]

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        
        #*** Insert your own code here and remove the following error message 
        
        S = np.zeros(tmax, dtype=np.int64)
        S[0] = DiscreteD(self.q).rand(1)  # Initial State S1
        
        # Infinite duration, length(S)=tmax
        if self.is_finite == False:
            
            for i in range(1, tmax):
                S[i] = DiscreteD(self.A[S[i-1]-1]).rand(1)
        
        # Finite duration, length(S)<=tmax
        else:
            
            for i in range(1, tmax):
                S[i] = DiscreteD(self.A[S[i-1]-1]).rand(1)
                if S[i] < self.A.shape[1]:
                    continue
                else:
                    tend = i
                    S = S[:tend]
                    break
            
        return S
    

    def viterbi(self):
        pass
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass

    def forward(self, pX):

        n_states = pX.shape[0]
        n_obsrvs = pX.shape[1]
        alpha_temp = np.zeros((n_states, n_obsrvs))
        alpha_hat = np.zeros((n_states, n_obsrvs))
        c = np.zeros(n_obsrvs)

        # Initialization
        alpha_temp[:, 0] = self.q * pX[:, 0]
        c[0] = np.sum(alpha_temp[:, 0])
        alpha_hat[:, 0] = alpha_temp[:, 0] / c[0]

        # Forward Step
        for t in range(1, n_obsrvs):
            alpha_temp[:, t] = pX[:, t] * np.dot(alpha_hat[:, t-1], self.A[:, :n_states])
            c[t] = np.sum(alpha_temp[:, t])
            alpha_hat[:, t] = alpha_temp[:, t] / c[t]

        # Termination
        if self.is_finite:
            c = np.append(c, np.dot(alpha_hat[:, -1], self.A[:, -1]))

        return alpha_hat, c


    def finiteDuration(self):
        pass
    
    def backward(self):
        pass

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
