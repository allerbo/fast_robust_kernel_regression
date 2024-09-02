import numpy as np

class prox_grad():
  def __init__(self, K,y,lbda, nrm, space='par', step_size=0.01):
    assert space in ['par', 'pred'], 'Non-valid space: '+space+'!'
    assert nrm in ['l1','l2','linf'], "Non-valid norm: "+nrm+'!'
    self.K=K
    self.y=y
    self.lbda=lbda
    self.nrm=nrm
    self.space=space
    self.step_size=step_size
    n=y.shape[0]
    if space=='par':
      self.var=np.zeros((n,1))
    elif space=='pred':
      ns=K.shape[0]
      self.var=np.zeros((ns,1))
      self.Ih=np.hstack((np.eye(n),np.zeros((n,ns-n))))
  
  def prox(self, x):
    if self.nrm=='l1':
      return np.sign(x)*np.maximum(np.abs(x)-self.lbda,0)
    if self.nrm=='l2':
      return x/(1+self.lbda)
    if self.nrm=='linf':
      return x-self.euclidean_proj_l1ball(np.squeeze(x),self.lbda).reshape((-1,1))
  
  def prox_step(self):
    grad=self.K@self.var-self.y if self.space=='par' else self.K@(self.Ih@self.var-self.y)
    var_g=self.var-self.step_size*grad
    self.var=self.prox(var_g)
  
  def get_fh(self):
    return self.var

  def get_alphah(self):
    return self.var
  
  
  #The code below is taken from https://gist.github.com/daien/1272551
  def euclidean_proj_simplex(self, v, s=1):
      assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
      n, = v.shape  # will raise ValueError if v is not 1-D
      # check if we are already on the simplex
      if v.sum() == s and np.alltrue(v >= 0):
          # best projection: itself!
          return v
      # get the array of cumulative sums of a sorted (decreasing) copy of v
      u = np.sort(v)[::-1]
      cssv = np.cumsum(u)
      # get the number of > 0 components of the optimal solution
      rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
      # compute the Lagrange multiplier associated to the simplex constraint
      theta = float(cssv[rho] - s) / (rho+1)
      # compute the projection by thresholding v using theta
      w = (v - theta).clip(min=0)
      return w
  
  
  def euclidean_proj_l1ball(self, v, s=1):
      """ Compute the Euclidean projection on a L1-ball
      Solves the optimisation problem (using the algorithm from [1]):
          min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
      Parameters
      ----------
      v: (n,) numpy array,
         n-dimensional vector to project
      s: int, optional, default: 1,
         radius of the L1-ball
      Returns
      -------
      w: (n,) numpy array,
         Euclidean projection of v on the L1-ball of radius s
      Notes
      -----
      Solves the problem by a reduction to the positive simplex case
      See also
      --------
      euclidean_proj_simplex
      """
      assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
      n, = v.shape  # will raise ValueError if v is not 1-D
      # compute the vector of absolute values
      u = np.abs(v)
      # check if v is already a solution
      if u.sum() <= s:
          # L1-norm is <= s
          return v
      # v is not already a solution: optimum lies on the boundary (norm == s)
      # project *u* on the simplex
      w = self.euclidean_proj_simplex(u, s=s)
      # compute the solution to the original problem on v
      w *= np.sign(v)
      return w
