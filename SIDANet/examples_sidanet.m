rng('default'); %for reproducibility

% This function depends on the CVX package. Please download at
%http://cvxr.com/cvx/download/


%load example data;
load '...\exampledata_sidanet.mat';

rng('default'); %for reproducibility
%perform classification and association 
gridMethod='RandomSearch';  % RandomSearch or GridSearch; 
doParallel='True'; %perfrom parallel computing. Use this for computation efficiency
%It will first check if the parallel computing toolbox is installed. If not, no parallel computing. 
withCov='False'; %True or False; If True, assumes last dataset has all covariates;
%use indicator matrix for binary or categorical variable. 
plotIt='False'; %True or False; plots discriminant and correlation plots if True
tic;
sidanet_cvout= sidanet_cvRandom(Xdata,Y,edged,vWeightd,withCov,plotIt,Xtestdata,Ytest,gridMethod,doParallel);
t11=toc; 
time1=t11/60;



rng('default'); %for reproducibility
tic;
doParallel='False';
sidanet_cvout= sidanet_cvRandom(Xdata,Y,edged,vWeightd,withCov,plotIt,Xtestdata,Ytest,gridMethod,doParallel);
t2=toc; 
time=t2/60;

 Xdata=cellfun(@(x) mynormalize(x), Xdata,'UniformOutput',false);
 Xtestdata=cellfun(@(x) mynormalize(x), Xtestdata,'UniformOutput',false);
 
 
%use optimal tuning parameter to obtain discriminant vectors and predict testing data;
optTau=sidanet_cvout.optTau;
[sida_error,sida_correlation,hatalpha,predclass]= sidanet(Xdata,Y,edged,vWeightd,optTau,withCov,Xtestdata,Ytest);


