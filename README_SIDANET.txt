Sparse Integrative Discriminant Analysis with Network data (SIDANet) README
Dec 27, 2019
1.This package depends on the CVX package. 
Please download CVX ( available at http://cvxr.com/cvx/download/ ), and ensure that both packages are added to your Matlab path library.


2. Please run examples_sidanet.m for examples. 


3. This Matlab package is for realizing the SIDANet algorithm proposed in the following paper.
Please cite this paper if you use the code for your research purpose.


Sparse Linear Discriminant Analysis for Multi-view Structured Data (2019). 

4.Please send your comments and bugs to seaddosafo@gmail.com

These are the main functions in this package:
%--------------------------------------------------------------------------
%sidanet.m: function to perform sparse integrative disdcriminant analysis  for fixed tuning parameters, when prior information is available. 
%Outputs integrative discriminant vectors, estimate misclassification rate, total correlation coefficient (RV coefficient), predicted class
%--------------------------------------------------------------------------

%DESCRIPTION:
%It is recommended to use sidanet_tunerange.m to obtain lower and upper bounds for 
%the tuning parameters since too large tuning parameters will result in 
%trivial solution vector (all zeros), and too small may result in
%non-sparse vectors. 

%--------------------------------------------------------------------------
%sidanet_cvRandom.m: cross validation approach to
%select optimal tuning parameters for sparse integrative discriminant
%analysis with network data. Allows for inclusion of covariates which are not penalized.
%--------------------------------------------------------------------------
%
%DESCRIPTION:
%Function performs nfolds cross validation to select
%optimal tuning parameter, which is then applied on whole data or testing data
% to predict class membership. 
%If you want to apply optimal tuning parameters to testing data, you may
%also use sidanet.m  Tuning parameters are optimized over full grid or random grid.
%            Default is RandomSearch.

%USAGE
%see examples.m on how to use it

%--------------------------------------------------------------------------
%integrativeDA.m: yields solution to Theorem 1, It depends on only covariance
%matrices. 
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%DiscriminantPlots.m: function to visualize separation of the classes.
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%CorrelationPlots.m: function to visualize strength of association of
%discriminant scores
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%sidanet_tunerange.m: function to provide tuning parameter lower and upper bounds
%to estimate canonical discriminant vectors;
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%sida_inner.m: %this calls the CVX algorithm to estimate canonical discriminant
%loadings
%it is called in sida_cvRandom.m and sida.m
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
mynormalize.m: function to normalize data to have mean zero and variance 1
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
%myfastIDAnonsparse.m: function to obtain nonsparse solution to integrative lda problem
%and to obtain matrix needed in constraints
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%sida_classify.m: function to perform classification with estimated
%discriminant loadings. Classification can be done with discriminant vectors from 
%all views combined or on separate views.
%--------------------------------------------------------------------------



%--------------------------------------------------------------------------
%examples_sidanet.m: You can do the following:
% 1. Can load example simulated data: exampledata_sidanet
% 2. Can use sidanet_cvRandom.m to obtain integrative discriminant vectors, estimated classificaition
%rate, predicted class, and estimated correlation.
% 3. For any fixed tuning parameter value, you can use sida.m to obtain
% integrative discriminant vectors, estimated classificaition
% 4. Can plot discriminant and correlation plots.
%--------------------------------------------------------------------------



