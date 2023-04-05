import numpy as np
import numpy.matlib
import scipy.stats as stats

class GaussD:
    """
    GaussD - Probability distribution class, representing
    Gaussian random vector
    EITHER with statistically independent components,
               i.e. diagonal covariance matrix, with zero correlations,
    OR with a full covariance matrix, including correlations
    -----------------------------------------------------------------------
    
    Several GaussD objects may be collected in a multidimensional array,
               even if they do not have the same DataSize.
    """
    def __init__(self, means, stdevs=None, cov=None):

        self.means = np.array(means)
        self.stdevs = np.array(stdevs)
        self.dataSize = len(self.means)

        if cov is None:
            self.variance = self.stdevs**2
            self.cov = np.eye(self.dataSize)*self.variance
            self.covEigen = 1
        else:
            self.cov = cov
            v, self.covEigen = np.linalg.eig(0.5*(cov + cov.T))
            self.stdevs = np.sqrt(np.abs(v))
            self.variance = self.stdevs**2
    
   
    def rand(self, nData):
        """
        R=rand(pD,nData) returns random vectors drawn from a single GaussD object.
        
        Input:
        pD=    the GaussD object
        nData= scalar defining number of wanted random data vectors
        
        Result:
        R= matrix with data vectors drawn from object pD
           size(R)== [length(pD.Mean), nData]
        """
        R = np.random.randn(self.dataSize, nData)
        R = np.diag(self.stdevs)@R
        
        if not isinstance(self.covEigen, int):
            R = self.covEigen@R

        R = R + np.matlib.repmat(self.means.reshape(-1, 1), 1, nData)

        return R
    
    def init(self):
        pass
    
    def likelihood(self, obsrv):

        pX = np.zeros(len(obsrv))
        for j in range(len(obsrv)):
            pX[j] = stats.norm.pdf(obsrv[j], self.means[0], self.stdevs[0])

        return pX
    
    def prob(obsrv, distr):

        pX = [d.likelihood(obsrv) for d in distr]
        pX = np.array(pX)

        pX_scaled = np.zeros(pX.shape)
        factors = np.zeros(pX.shape[1])
        for i in range(pX_scaled.shape[0]):
            for j in range(pX_scaled.shape[1]):
                pX_scaled[i, j] = pX[i, j] / np.amax(pX[:, j])
                factors[j] = np.amax(pX[:, j])

        return pX, pX_scaled, factors

    def logprob(pDs, x):
        pass
    
    def plotCross(self):
        pass

    def adaptStart(self):
        pass
    
    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass