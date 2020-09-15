function [sida_error,sida_correlation,hatalpha,predclass]= sida(Xdata,Y,Tau,withCov,Xtestdata,Ytest,AssignClassMethod,plotIt,standardize, maxiteration,weight)

%--------------------------------------------------------------------------
%sida.m: function to perform sparse integrative disdcriminant analysis,
%estimate misclassification rate, total correlation coefficient
%
%--------------------------------------------------------------------------
%
%DESCRIPTION:
%For fixed tuning parameter for each dataset in Xdata, and a vector Y of class membership
%function performs sparse integrative discriminant analysis.
%It then outputs classification error on the training data (Xdata) or
%testing data (Xtestdata).
%It is recommended to use sida_tunerange.m to obtain lower and upper bounds for 
%the tuning parameters since too large tuning parameter will result in 
%trivial solution vector (all zeros), and too small may result in
%non-sparse vectors. 

%USAGE
%[sida_error,hatalpha,predclass]= sida(Xdata,Y,Tau,Xtestdata,Ytest,AssignClassMethod,standardize, maxiteration,weight);;
%[sida_error,hatalpha,predclass]= sida(Xdata,Y,Tau,Xtestdata,Ytest) ;%uses defualt settings
%[sida_error,hatalpha,predclass]= sida(Xdata,Y,Tau) %assumes no testing
%data. Predicts classification error on Xdata. 
%see examples.m for examples


%DATE: March 08,2019
%
%AUTHORS
%MATLAB CODE WAS WRITTEN BY SANDRA E. SAFO (seaddosafo@gmail.com)
%

% This function depends on the CVX package. Please download at
%http://cvxr.com/cvx/download/


%%%%Inputs
%Xdata:     1 by d cell array with each cell containing the n by p_d dataset
%Y:         n x 1 vector of class membership
%Tau:       d by 1 vector of tuning parameter. It is recommended to use 
%           sida_tunerange.m to obtain lower and upper bounds for 
%           the tuning parameters since too large tuning parameter will result in 
%           trivial solution vector (all zeros), and too small may result in
%           non-sparse vectors. 
%withCov    True or False if covariates are available. If True, please set
%           all covariates as one dataset and should be the last dataset.
%           For binary and categorical variables, use indicator
%           matrices/vectors. Default is False. 
%Xtestdata:  1 by d cell array with each cell containing the n by p_d
%            dataset.   Use if you want to predict on a testing dataset.
%Ytest:      n x 1 vector of class membership
%AssignClassMethod: classification method. Either Joint or Separate. Joint
%uses all discriminant vectors from D datasets to predict class membership
%Separate predicts class membership separately for each dataset. Default is
%Joint.
%plotIt:    True or False; produces discriminants and correlation plots.
%           Default is 'False';
%standardize: True or False. If True, data will be normalized to have mean 0
%                   and variance one for each variable. Default is True.
%maxiteration: maximum iteration
%weight: balances separation and association. Default is 0.5.

%Output
%sida_error- estimated classication error
%sida_correlation= sum of correlations. Normalized so that the maximum is
%one. 
%hatlapha- a 1 x d cell of  estimated canonical discriminant vectors for each dataset
%predclass- a ntest by 1 vector of predicted class

%set defaults;
narginchk(3,11);
if(nargin <4)
    withCov='False';
    Xtestdata=Xdata;
    Ytest= Y;
    AssignClassMethod='Joint';
    plotIt='False';
    standardize='True';
    maxiteration=10;
    weight=0.5;
end
if(nargin <5)
    Xtestdata=Xdata;
    Ytest= Y;
    AssignClassMethod='Joint';
    plotIt='False';
    standardize='True';
    maxiteration=10;
    weight=0.5;
end

if(nargin <7)
    AssignClassMethod='Joint';
    plotIt='False';
    standardize='True';
    maxiteration=10;
    weight=0.5;
end

if(nargin <8)
    plotIt='False';
    standardize='True';
    maxiteration=10;
    weight=0.5;
end


if(nargin <9)
    standardize='True';
    maxiteration=10;
    weight=0.5;
end

if(nargin <10)
    maxiteration=10;
    weight=0.5;
end

if(nargin <11)
   weight=0.5;
end


dsizes = cellfun('size',Xdata,2);
D=length(dsizes);
nsizes=cellfun('size',Xdata,1);

if (all(nsizes(:)~=nsizes(1)))
    error('The datasets  have different number of observations');
end

if strcmp(standardize,'True')
    Xdata=cellfun(@(x) mynormalize(x), Xdata,'UniformOutput',false);
    Xtestdata=cellfun(@(x) mynormalize(x), Xtestdata,'UniformOutput',false);
end

%obtain sizes
[a,b]=size(Ytest);
if(b>a)
    Ytest=Ytest';
end

nK=length(unique(Y))-1;


%initialize
thresh=10^(-3);

    iter=0;
    diffalpha=1;
    reldiff=1;
    %diffalphaold=1;
    %obtain tildealpha and tildebeta which are nosparse solution to integrative LDA
    [tildealpha,tildelambda, myalpha,sqrtminv]=myfastIDAnonsparse(Xdata, Y,weight);

    %while convergence is not met
     while ( (iter<maxiteration) && (min(reldiff,max(diffalpha))>thresh)) %iterate if iteration is less than maxiteration or convergence not reached
          iter=iter+1;     
         fprintf('Current iteration is #%d\n', iter)
           myalphaold=myalpha;
          myalpha= sida_inner(Xdata,Y, sqrtminv,myalphaold,tildealpha,tildelambda,Tau,weight,withCov);

                 if( cellfun( @(myx) sum(sum(abs(myx)))==0,myalpha))
                    break
                end 
%                 mydiff=cellfun( @(xold,xnew) abs(xnew)-abs(xold), myalphaold,myalpha, 'UniformOutput',false); %use UniformOutput false if want to reterun vectors/matrices
%                   diffalpha=cell2mat(cellfun( @(x) norm(x,inf),mydiff,'UniformOutput',false)); %infinite norm of each direction vector for each D
                mydiff=cellfun( @(xold,xnew) xnew-xold, myalphaold,myalpha, 'UniformOutput',false); %use UniformOutput false if want to reterun vectors/matrices
                diffalpha=cell2mat(cellfun( @(xdiff,xold) norm(xdiff,'fro')^2/norm(xold,'fro')^2,mydiff,myalphaold,'UniformOutput',false)); %infinite norm of each direction vector for each D
                sumnormdiff=sum(cell2mat(cellfun( @(xdiff,xold) norm(xdiff,'fro')^2 ,mydiff,myalphaold,'UniformOutput',false))); %infinite norm of each direction vector for each D
                sumnormold=sum(cell2mat(cellfun( @(xdiff,xold) norm(xold,'fro')^2 ,mydiff,myalphaold,'UniformOutput',false)));
                reldiff=sumnormdiff/sumnormold;
     end
     myalphamat{1,1}=myalpha;

myalphamat2=vertcat(myalphamat{:})'; %unpacks 
dsizes = cellfun('size',Xdata,2);
hatalpha=(mat2cell(cell2mat(myalphamat2),dsizes,nK))';   
     
Projtest=cellfun(@(x,y) x*y, Xtestdata,hatalpha,'UniformOutput',false );
Projtrain=cellfun(@(x,y) x*y, Xdata,hatalpha,'UniformOutput',false );

predclass = sida_classify(Projtest, Projtrain, Y,AssignClassMethod); 
if(strcmp(AssignClassMethod,'Joint'))
     sida_error=sum(predclass~=Ytest)/length(Ytest);
elseif(strcmp(AssignClassMethod,'Separate'))
    for d=1:D
     sida_error(1,d)= sum(predclass(:,d)~=Ytest)/length(Ytest); 
    end
end
%sum pairwise RV correlations
 for d=1:D
        dd=setdiff(1:D,d);
        %cross-covariance
        sumCorr2=0;
        for j=dd;
        Xdd=Xtestdata{1,d}*hatalpha{1,d};
        Xjj=Xtestdata{1,j}*hatalpha{1,j};
        Sdd= cov(Xdd);
        Sjj=cov(Xjj);
        %center matrices
        cSd=Xdd-repmat(mean(Xdd),size(Xdd,1),1);
        cSj=Xjj-repmat(mean(Xjj),size(Xjj,1),1);
        Sdj=cSd'*cSj/(size(cSd,1)-1);
        sumcorr3=trace(Sdj*Sdj')/(trace(Sdd*Sdd')^.5*trace(Sjj*Sjj')^0.5);
        sumCorr2=sumCorr2+sumcorr3;
        end
        ss(d,:)=sumCorr2/length(dd);
 end
 
sida_correlation=sum(ss)/D;

%Generate discriminant and correlation plots if plotIt is 'True';
if strcmp(plotIt,'True')
    plotoutCorr=CorrelationPlots(Xtestdata,Ytest,hatalpha); %correlation plots;
    plotoutDisc=DiscriminantPlots(Xtestdata,Ytest,hatalpha);
end



%display some results
% NonZeroAlpha=cell2mat(cellfun(@(x) sum(x~=0), hatalpha','UniformOutput',false ));
% NonZeroAlpha=NonZeroAlpha';
% dNZalpha=['Number of non-zeros alpha for each D: ', num2str(NonZeroAlpha)];
% disp(dNZalpha);
dcorr=['Sum of Correlations: ', num2str(sida_correlation)];
disp(dcorr);
derror=['Estimated classification error: ', num2str(sida_error)];
disp(derror);
d=['SIDA classification method is: ', AssignClassMethod];
disp(d);
