
function myhatalpha= sida_inner(Xdata,Y, sqrtminv,myalphaold,tildealpha,  tildelambda, Tau,weight,withCov);
%Author:Sandra E. Safo
%DATE: April 30, 2019

%--------------------------------------------------------------------------
%sida_inner.m: %this calls the CVX algorithm to estimate canonical discriminant
%loadings
%it is called in sida_cvRandom.m and sida.m
%--------------------------------------------------------------------------

%Input
%Xdata is a a 1 by D cell with each cell containing the data types
%Tau is a D by 1 vector of tuning parameters
%tildelambda is nk by d vector;

%Output
%myalphamat- a 1 by D cell of estimated canonical discriminant vectors

% This function depends on the CVX package. Please download at
%http://cvxr.com/cvx/download/

[a,b]=size(Xdata);
D=max(a,b);
if(a>b)
  Xdata=Xdata';
end

nK=length(unique(Y))-1;
SepAndAssocd=myfastinner2(Xdata,Y, sqrtminv, myalphaold,tildealpha, weight);

myhatalpha=cell(1,D);

if(strcmp(withCov,'True'))
   Tau(D)=0.00001;
end

for d=1:D
    p=size(Xdata{1,d},2);
   
    %solve for alpha 
    cvx_begin quiet
        variable alphai(p,nK)
        
         minimize(sum(norms(alphai,2,2)))
         subject to
        % if(and((d==D),strcmp(withCov,'True'))) %last dataset is covariates
        % norm(SepAndAssocd{1,d} - alphai*diag(tildelambda(:,d)), Inf)<=0.00001;
         %else
         norm(SepAndAssocd{1,d} - alphai*diag(tildelambda(:,d)), Inf)<=Tau(d);
        % end
    cvx_end   
   
     %normalize alpha
        alphai(abs(alphai)<=10^-5)=0;
        if (min(sum(abs(alphai),1))==0)
            myalpha=alphai;
        else
        %hatalpha=alphai./repmat(sqrt(sum(alphai.*alphai)),p,1);
        [Q,~]=qr(alphai);
        myalpha=Q(:,1:nK);
        myalpha(abs(myalpha)<=10^-5)=0;
        end;
        myhatalpha{1,d}=myalpha;
end

clearvars alphai 