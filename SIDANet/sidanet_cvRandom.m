function sidanet_cvout= sidanet_cvRandom(Xdata,Y,edges,vweight,withCov,plotIt,Xtestdata,Ytest,gridMethod,doParallel,AssignClassMethod,nfolds,ngrid,standardize,maxiteration, weight,eta);



%perform 5 fold cross validation;
%--------------------------------------------------------------------------
%sidanet_cvRandom.m: cross validation approach to
%select optimal tuning parameters for sparse integrative discriminant
%analysis with network data
%--------------------------------------------------------------------------
%
%DESCRIPTION:
%Function performs nfolds cross validation to select
%optimal tuning parameter, which is then applied on whole data or testing data
% to predict class membership. 
%If you want to apply optimal tuning parameters to testing data, you may
%also use sida.m


%USAGE
%sida_cvout=sidanet_cvRandom(Xdata,Y)% uses default settings
%%sida_cvout= sidanet_cvRandom(Xdata,Y,Xtestdata,Ytest) %if you want to
%%predict on a testing dataset
%sida_cvout= sidanet_cvRandom(Xdata,Y,Xtestdata,Ytest,gridMethod,doParallel,AssignClassMethod,nfolds,ngrid,standardize,maxiteration, weight);
%you can omit any of the options.
%see examples.m for examples

%DATE: May 6, 2019
%
%AUTHORS
%MATLAB CODE WAS WRITTEN BY SANDRA E. SAFO (seaddosafo@gmail.com)
%
%REFERENCES
%Sparse Discriminant Analysis for Multi-viewStructured Data- Sandra E.
%Safo, Eun Jeong Min, and Lillian Haine 2019

% This function depends on the CVX package. Please download at
%http://cvxr.com/cvx/download/


%DATE: May 6, 2019
%
%AUTHORS
%MATLAB CODE WAS WRITTEN BY SANDRA E. SAFO (seaddosafo@gmail.com)
%
%REFERENCES

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
%gridMethod: Optimize tuning parameters over full grid or random grid.
%Default is RandomSearch. 
%doParallel: Yes or No for parallel computing. Default is yes if parallel
%computing tool box is available.
%AssignClassMethod: classification method. Either Joint or Separate. Joint
%uses all discriminant vectors from D datasets to predict class membership
%Separate predicts class membership separately for each dataset. Default is
%Joint.
%eta         is a scalar between 0 and 1 that balances smoothing penalty and sparse
%            penalty. Default is 0.5;
%edges       is a 1 by D cell data that contains edge information (M x 2, M is number of edges) for each data.
%            if no edge information, set as [] (empty). This will use sida to obtain
%            sparse canonical variates;
%vweight     is a 1 by D cell data for the weight of the vertices
%(variables) for each D.

%nfolds: number of cross validation folds. Default is 5.
%ngrid: Number of grid points for tuning parameters. Default is 8 if D=2
%If D>2, default is 5;
%standardize: True or False. If True, data will be normalized to have mean 0
%                   and variance one for each variable. Default is True.
%maxiteration: maximum iteration
%weight: balances separation and association. Default is 0.5.


%output
%sidanet_cvout.hatalpha:           estimated canonical discriminant vectors
%sidanet_cvout.sida_correlation:   estimated total correlation;
%sidanet_cvout.predclass:          predicted Class;
%sidanet_cvout.sidaerror:          estimated error rate;
%sidanet_cvout.optTau:             optimal tuning parameters;
%sidanet_cvout.gridpoints:         grid points used;


%set defaults;
narginchk(4,17)
if(nargin <5)
    withCov='False';
    plotIt='False';
    Xtestdata=Xdata;
    Ytest= Y;
    gridMethod='RandomSearch';
    doParallel='True';
    AssignClassMethod='Joint';
    nfolds=5;
    ngrid=8;
    standardize='True';
    maxiteration=10;
    weight=0.5;
    eta=0.5;
end

if(nargin <6)
    plotIt='False';
    Xtestdata=Xdata;
    Ytest= Y;
    gridMethod='RandomSearch';
    doParallel='True';
    AssignClassMethod='Joint';
    nfolds=5;
    ngrid=8;
    standardize='True';
    maxiteration=10;
    weight=0.5;
    eta=0.5;
end



if(nargin <7)
    Xtestdata=Xdata;
    Ytest= Y;
    gridMethod='RandomSearch';
    doParallel='True';
    AssignClassMethod='Joint';
    nfolds=5;
    ngrid=8;
    standardize='True';
    maxiteration=10;
    weight=0.5;
    eta=0.5;
end

if(nargin <9)
    gridMethod='RandomSearch';
    doParallel='True';
    AssignClassMethod='Joint';
    nfolds=5;
    ngrid=8;
    standardize='True';
    maxiteration=10;
    weight=0.5;
    eta=0.5;
end

if(nargin <10)
    doParallel='True';
    AssignClassMethod='Joint';
    nfolds=5;
    ngrid=8;
    standardize='True';
    maxiteration=10;
    weight=0.5;
    eta=0.5;
end

if(nargin <11)
    AssignClassMethod='Joint';
    nfolds=5;
    ngrid=8;
    standardize='True';
    maxiteration=10;
    weight=0.5;
    eta=0.5;
end

if(nargin <12)
    nfolds=5;
    ngrid=8;
    standardize='True';
    maxiteration=10;
    weight=0.5;
    eta=0.5;
end

if(nargin <13)
    ngrid=8;
    standardize='True';
    maxiteration=10;
    weight=0.5;
    eta=0.5;
end

if(nargin <14)
    standardize='True';
    maxiteration=10;
    weight=0.5;
    eta=0.5;
end

if(nargin <15)
    maxiteration=10;
    weight=0.5;
    eta=0.5;
end

if(nargin <16)
    weight=0.5;
    eta=0.5;
end
if(nargin <17)
    eta=0.5;
end


dsizes = cellfun('size',Xdata,2);
D=length(dsizes);
nsizes=cellfun('size',Xdata,1);

if (D==1)
    error('There should be at least two datasets');
end

if (all(nsizes(:)~=nsizes(1)))
    error('The datasets  have different number of observations');
end

%check if parallel computing toolbox is installed
isParallel=license('test','Distrib_Computing_Toolbox');


if strcmp(standardize,'True')
    Xdata=cellfun(@(x) mynormalize(x), Xdata,'UniformOutput',false);
    Xtestdata=cellfun(@(x) mynormalize(x), Xtestdata,'UniformOutput',false);
end




[a,b]=size(Y);
if(b>a)
    Y=Y';
end
[a,b]=size(Ytest);
if(b>a)
    Ytest=Ytest';
end

%randomly choose positions from the hyperspace
%allows each canonical discriminant vector to have different tuning paramter range
rng('default');
class=unique(Y);
nc=length(class);
nK=nc-1;

%obtain tuning range common to all K
%tic;
if(strcmp(withCov,'True'))
    Dnew=D-1;
else
    Dnew=D;
end
if Dnew>2
    ngrid=5;
end
Tauvec= sidanet_tunerange(Xdata,Y,ngrid,standardize,weight,eta, edges,vweight,withCov);
%calculate the normalized laplacian
myL =myNLaplcianG(Xdata,edges,vweight);
%toc
%

Nn=zeros(nc,1);
foldid3=[];
 for i=1:nc;
    Nn(i)=sum(Y==i);
    foldid3=[foldid3 randsample([repmat(1:nfolds,1,floor(Nn(i)/nfolds)) 1:mod(Nn(i),nfolds)],Nn(i));];
 end 
 
 %define the grid;

 Tauvec2=Tauvec{1,1}';  
 combin=cell(1, numel(Tauvec2)); %cell array with D vectors to combine
 [combin{:}] = ndgrid(Tauvec2{:});
 combin = cellfun(@(x) x(:), combin,'uniformoutput',false); 
 mygrid = [combin{:}]; 
 mygridm{1,1}=mygrid;


gridcomb=size(mygridm{1,1},1);
 if(strcmp(gridMethod,'RandomSearch'))
     if Dnew==2
     ntrials=floor(0.2*gridcomb);
     elseif Dnew>2
     ntrials=floor(0.15*gridcomb);
     end
     mytune=randsample(1:gridcomb,ntrials);
     Tau=cellfun(@(x) x(mytune,:),mygridm,'uniformoutput',false);
 elseif(strcmp(gridMethod,'GridSearch'))
     ntrials=size(mygrid,1);
     Tau={mygrid};
 end

 
ErrTau=[];
tic
ticBytes(gcp);
  Xtrain=Xdata(:,1:D); 
  Ytrain={Y(:,1)};
if (and(strcmp(doParallel,'True'),isParallel==1)) %parallel computing if yes and Parallel computing  toolbox is installed. 
     parfor itau=1:ntrials
         Taux=(cell2mat(cellfun(@(x) x(itau,:), Tau, 'uniformoutput',false)))';      
         for r=1:nfolds
              which=foldid3==r;   
              Xtrain2=cellfun(@(x) x(~which,:),Xtrain,'uniformoutput',false);
              Xtest2=cellfun(@(x) x(which,:),Xtrain,'uniformoutput',false);
              Ytrain2=cell2mat(cellfun(@(y) y(~which,1),Ytrain,'uniformoutput',false));
              Ytest2=cell2mat(cellfun(@(y) y(which,1),Ytrain,'uniformoutput',false));
              [sida_error,~,~,~]= sidanet(Xtrain2,Ytrain2,edges,vweight,Taux,withCov,Xtest2,Ytest2,myL,AssignClassMethod,standardize, maxiteration,weight,eta);
              ErrTau=[ErrTau;[{Taux} itau r sida_error ]];
         end
     end  
else
for itau=1:ntrials
         Taux=(cell2mat(cellfun(@(x) x(itau,:), Tau, 'uniformoutput',false)))';      
         for r=1:nfolds
              which=foldid3==r;   
              Xtrain2=cellfun(@(x) x(~which,:),Xtrain,'uniformoutput',false);
              Xtest2=cellfun(@(x) x(which,:),Xtrain,'uniformoutput',false);
              Ytrain2=cell2mat(cellfun(@(y) y(~which,1),Ytrain,'uniformoutput',false));
              Ytest2=cell2mat(cellfun(@(y) y(which,1),Ytrain,'uniformoutput',false));
              [sida_error,~,~,~]= sidanet(Xtrain2,Ytrain2,edges,vweight,Taux,withCov,Xtest2,Ytest2,myL,AssignClassMethod,standardize, maxiteration,weight,eta);
              ErrTau=[ErrTau;[{Taux} itau r sida_error ]];
         end
 end 
end
tocBytes(gcp)
toc


ErrTau3=ErrTau;
ErrTau3(:,1)=[];
ErrTau3=cell2mat(ErrTau3);
mErr=round(grpstats(min(ErrTau3(:,3:end),[],2),ErrTau3(:,1)),4);


[row,~]=find(mErr(:,1)==min(mErr(:,1)),1,'last');
ind=find((ErrTau3(:,1)==row),1,'first');
optTau=cell2mat(ErrTau(ind,1));

%Apply on testing data;
[sida_error,~,hatalpha,predclass]=sidanet(Xdata,Y,edges,vweight,optTau,withCov,Xtestdata,Ytest,myL,AssignClassMethod,standardize, maxiteration,weight,eta);

%sum pairwise correlations
  %sum pairwise RV coefficients
 for d=1:D
        dd=setdiff(1:D,d);
        Xdd=Xtestdata{1,d}*hatalpha{1,d};
        %cross-covariance
        sumCorr2=0;
        for j=dd
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
%display some results
dcorr=['RV Coefficient: ', num2str(sida_correlation)];
disp(dcorr);
derror=['Estimated misclassification error: ', num2str(sida_error)];
disp(derror);
d=['SIDA classification method is: ', AssignClassMethod];
disp(d);

sidanet_cvout.hatalpha=hatalpha;
sidanet_cvout.sida_correlation=sida_correlation;
sidanet_cvout.predclass=predclass;
sidanet_cvout.sidaerror=sida_error;
sidanet_cvout.optTau=optTau;
sidanet_cvout.gridpoints=Tau;




