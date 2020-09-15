function [sida_error,sida_correlation,hatalpha,predclass]= sidanet(Xdata,Y,edges,vweight,Tau,withCov,Xtestdata,Ytest,myL,AssignClassMethod,standardize, maxiteration,weight,eta);

%--------------------------------------------------------------------------
%sidneta.m: function to perform sparse integrative disdcriminant analysis,
%estimate misclassification rate, rv coefficient
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
%[sida_error,hatalpha,predclass]= sidanet(Xdata,Y,edges,vweight,Tau,withCov,Xtestdata,Ytest,AssignClassMethod,standardize, maxiteration,weight,eta);
%[sida_error,hatalpha,predclass]= sidanet(Xdata,Y,edges,vweight,Tau,withCov,Xtestdata,Ytest) ;%uses defualt settings
%[sida_error,hatalpha,predclass]= sidanet(Xdata,Y,edges,vweight,Tau,withCov)
%%assumes no covariates, no testing data
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
%Xtestdata:  1 by d cell array with each cell containing the n by p_d
%            dataset.   Use if you want to predict on a testing dataset.
%Ytest:      n x 1 vector of class membership
%AssignClassMethod: classification method. Either Joint or Separate. Joint
%uses all discriminant vectors from D datasets to predict class membership
%Separate predicts class membership separately for each dataset. Default is
%Joint.%standardize: True or False. If True, data will be normalized to have mean 0
%             and variance one for each variable. Default is True.
%eta         is a scalar between 0 and 1 that balances smoothing penalty and sparse
%            penalty. Default is 0.5;
%edges       is a 1 by D cell data that contains edge information (M x 2, M is number of edges) for each data.
%            if no edge information, set as [] (empty). This will use sida to obtain
%            sparse canonical variates;
%vweight     is a 1 by D cell data for the weight of the vertices
%(variables) for each D.


%Output
%sida_error- estimated classication error
%sida_correlation= sum of correlations. Normalized so that the maximum is
%one. 
%hatlapha- a 1 x d cell of  estimated canonical discriminant vectors for each dataset
%predclass- a ntest by 1 vector of predicted class

%set defaults;
narginchk(5,14);
if(nargin <6)
    withCov='False';
    Xtestdata=Xdata;
    Ytest= Y;
    %Tau=[];
    myL=myNLaplcianG(Xdata,edges,vweight);
    AssignClassMethod='Joint';
    standardize='True';
    maxiteration=10;
    weight=0.5;
    eta=0.5;
end

if(nargin <9)
    myL=myNLaplcianG(Xdata,edges,vweight);
    AssignClassMethod='Joint';
    standardize='True';
    maxiteration=10;
    weight=0.5;
    eta=0.5;
end
if(nargin <10)
    if(isempty(myL))
        myL=myNLaplcianG(Xdata,edges,vweight);
    end
    AssignClassMethod='Joint';
    standardize='True';
    maxiteration=10;
    weight=0.5;
    eta=0.5;
end
if(nargin <11)
    if(isempty(myL))
        myL=myNLaplcianG(Xdata,edges,vweight);
    end
    standardize='True';
    maxiteration=10;
    weight=0.5;
    eta=0.5;
end

if(nargin <12)
    if(isempty(myL))
        myL=myNLaplcianG(Xdata,edges,vweight);
    end
    maxiteration=10;
    weight=0.5;
    eta=0.5;
end

if(nargin <13)
    if(isempty(myL))
        myL=myNLaplcianG(Xdata,edges,vweight);
    end
    weight=0.5;
    eta=0.5;
end

if(nargin <14)
    if(isempty(myL))
        myL=myNLaplcianG(Xdata,edges,vweight);
    end
   eta=0.5;
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


%obtain tuning range common to all K
%tic;
%[TauX1range, TauX2range]= sida_tunerange(X1,X2,Y,ngrid,standardize);
%toc;

%Taux1=TauX1range(:,2);
%Taux2=TauX2range(:,2);


%initialize
thresh=10^(-3);

    iter=0;
    diffalpha=1;
    reldiff=1;
    relObj=1;
    %diffalphaold=1;
    %obtain tildealpha and tildebeta which are nosparse solution to integrative LDA
    [tildealpha,tildelambda, myalpha,sqrtminv]=myfastIDAnonsparse(Xdata, Y,weight);

    %while convergence is not met
     while ( (iter<maxiteration) && (min(min(reldiff,relObj),max(diffalpha))>thresh)) %iterate if iteration is less than maxiteration or convergence not reached
          iter=iter+1;     
         fprintf('Current iteration is #%d\n', iter)
           myalphaold=myalpha;
          [myalpha,myL]= sidanet_inner(Xdata,Y, sqrtminv,myalphaold,tildealpha,tildelambda,Tau,weight,eta, edges,vweight,withCov,myL);

                if( cellfun( @(myx) sum(sum(abs(myx)))==0,myalpha))
                    break
                end 
%                 mydiff=cellfun( @(xold,xnew) abs(xold)-abs(xnew), myalphaold, myalpha,'UniformOutput',false); %use UniformOutput false if want to reterun vectors/matrices
%                 diffalpha=cell2mat(cellfun( @(x) norm(x,inf),mydiff,'UniformOutput',false)); %infinite norm of each direction vector for each D  
                 mydiff=cellfun( @(xold,xnew) xnew-xold, myalphaold,myalpha, 'UniformOutput',false); %use UniformOutput false if want to reterun vectors/matrices
                diffalpha=cell2mat(cellfun( @(xdiff,xold) norm(xdiff,'fro')^2/norm(xold,'fro')^2,mydiff,myalphaold,'UniformOutput',false)); %infinite norm of each direction vector for each D
                ObjNew=sum(cell2mat(cellfun( @(xnew,Lp) (1-eta)*norm(xnew,'fro')^2 + eta*norm(Lp*xnew,'fro')^2 ,myalpha,myL,'UniformOutput',false)));
                ObjOld=sum(cell2mat(cellfun( @(xold,Lp) (1-eta)*norm(xold,'fro')^2 + eta*norm(Lp*xold,'fro')^2 ,myalphaold,myL,'UniformOutput',false)));
                relObj=abs(ObjNew-ObjOld)/ObjOld;
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
%sum pairwise RV correlations training data
 for d=1:D
        dd=setdiff(1:D,d);
        %cross-covariance
        sumCorr2=0;
        for j=dd
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

%display some results
dcorr=['RV Coefficient: ', num2str(sida_correlation)];
disp(dcorr);
derror=['Estimated misclassification error: ', num2str(sida_error)];
disp(derror);
d=['SIDA classification method is: ', AssignClassMethod];
disp(d);
