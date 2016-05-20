# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:17:45 2016

@author: Zero
"""

## Use multiple imputation to estimate the correlation coefficient and
## standard error between columns 1 and 2.

## Columns of interest
X = Z[:,1:3]

## Missing data patterns
ioo = np.flatnonzero(np.isfinite(X).all(1))
iom = np.flatnonzero(np.isfinite(X[:,0]) & np.isnan(X[:,1]))
imo = np.flatnonzero(np.isnan(X[:,0]) & np.isfinite(X[:,1]))
imm = np.flatnonzero(np.isnan(X).all(1))

## Complete data
XC = X[ioo,:]

## Number of multiple imputation iterations
nmi = 20

## Do the multiple imputation
F = np.zeros(nmi, dtype=np.float64)
for j in range(nmi):

    ## Bootstrap the complete data
    ii = np.random.randint(0, len(ioo), len(ioo))
    XB = XC[ii,:]

    ## Column-wise means
    X_mean = XB.mean(0)

    ## Column-wise standard deviations
    X_sd = XB.std(0)

    ## Correlation coefficient
    r = np.corrcoef(XB.T)[0,1]

    ## The imputed data
    XI = X.copy()

    ## Impute the completely missing rows
    Q = np.random.normal(size=(X.shape[0],2))
    Q[:,1] = r*Q[:,0] + np.sqrt(1 - r**2)*Q[:,1]
    Q = Q*X_sd + X_mean
    XI[imm,:] = Q[imm,:]

    ## Impute the rows with missing first column
    ## using the conditional distribution
    va = X_sd[0]**2 - r**2/X_sd[1]**2
    XI[imo,0] = r*X[imo,1]*(X_sd[0]/X_sd[1]) +\
                np.sqrt(va)*np.random.normal(size=len(imo))

    ## Impute the rows with missing second column
    ## using the conditional distribution
    va = X_sd[1]**2 - r**2/X_sd[0]**2
    XI[iom,1] = r*X[iom,0]*(X_sd[1]/X_sd[0]) +\
                np.sqrt(va)*np.random.normal(size=len(iom))

    ## The correlation coefficient of the imputed data
    r = np.corrcoef(XI[:,0], XI[:,1])[0,1]

    ## The Fisher-transformed correlation coefficient
    F[j] = 0.5*np.log((1+r) / (1-r))

## Apply the combining rule, see, e.g.
## http://sites.stat.psu.edu/~jls/mifaq.html#howto
FM = F.mean()
RM = (np.exp(2*FM)-1) / (np.exp(2*FM)+1)
VA = (1 + 1/float(nmi))*F.var() + 1/float(Z.shape[0]-3)
SE = np.sqrt(VA)
LCL,UCL = FM-2*SE,FM+2*SE
LCL = (np.exp(2*LCL)-1) / (np.exp(2*LCL)+1)
UCL = (np.exp(2*UCL)-1) / (np.exp(2*UCL)+1)

print "\nMultiple imputation:"
print "%.2f(%.2f,%.2f)" % (RM, LCL, UCL)