%Nov 13, 2019
rng('default'); %for reproducibility

% This function depends on the CVX package. Please download at
%http://cvxr.com/cvx/download/


%load example data;
%or you can simulate data BinaryExampleTable3.m or files
load 'mypath to \exampledata_sida.mat';

%perform association and classification simultaneously
gridMethod='RandomSearch';  % RandomSearch or GridSearch; 
doParallel='True'; %performs parallel computing. Use this for computation efficiency
withCov='False'; %True if covariates are available. If so, it should be the last dataset. %No regularization for covariates;
plotIt='True'; %Default is 'False'; Produces discriminant and correlation plots if True. 
%It will first check if the parallel computing toolbox is installed. If not, no parallel computing. 
tic;
sida_cvout= sida_cvRandom(Xdata,Y,withCov,plotIt,Xtestdata,Ytest,gridMethod,doParallel);
t1=toc; 
time=t1/60 


%No parallel computing
rng('default'); %for reproducibility
tic;
doParallel='False';
sida_cvout2= sida_cvRandom(Xdata,Y,withCov,plotIt,Xtestdata,Ytest,gridMethod,doParallel);
t2=toc; 
time=t2/60

%classify with optimal tuning parameters.
Xdata=cellfun(@(x) mynormalize(x), Xdata,'UniformOutput',false);
Xtestdata=cellfun(@(x) mynormalize(x), Xtestdata,'UniformOutput',false);
%use optimal tuning parameter to obtain discriminant vectors and predict testing data;
optTau=sida_cvout.optTau;
AssignClassMethod='Joint';
plotIt='True';
[sida_error,sida_correlation,hatalpha,predclass]= sida(Xdata,Y,optTau,withCov,Xtestdata,Ytest,AssignClassMethod,plotIt);


%Separate class assignments
AssignClassMethod='Separate';
plotIt='True';
[sida_error1,sida_correlation1,hatalpha1,predclass1]= sida(Xdata,Y,optTau,withCov,Xtestdata,Ytest,AssignClassMethod,plotIt);
