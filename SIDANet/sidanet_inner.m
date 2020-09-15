
function [myhatalpha,myL]= sidanet_inner(Xdata,Y, sqrtminv,myalphaold,tildealpha,  tildelambda, Tau,weight, eta, edges,vweight,withCov,myL);
%Author:Sandra E. Safo
%DATE: April 30, 2019

%--------------------------------------------------------------------------
%sida.m: %this calls the CVX algorithm to estimate canonical discriminant
%loadings
%it is called in cvsida.m and sida.m
%--------------------------------------------------------------------------

%Input
%Xdata is a  1 by D cell data with each cell containing the data types
%Tau is a D by 1 vector of tuning parameters
%tildelambda is nk by 1 vector;
%eta is a scalar between 0 and 1 that balances smoothing penalty and sparse
%penalty
%edges is a 1 by D cell data that contains edge information (M x 2, M is number of edges) for each data.
%if no edge information, set as [] (empty). This will use sida to obtain
%sparse canonical variates;
%vweight is a 1 by D cell data for the weight of the variables for each
%data type. Each entry is a p_d by 1 vecor, where p_d is the dimension for
%the dth dataset. 

%Output
%myalphamat- a 1 by D cell of estimated canonical discriminant vectors
% This function depends on the CVX package. Please download at
%http://cvxr.com/cvx/download/

if(nargin <13)
    myL =myNLaplcianG(Xdata,edges,vweight);
end

[a,b]=size(Xdata);
D=max(a,b);
if(a>b)
  Xdata=Xdata';
end

nK=length(unique(Y))-1;
SepAndAssocd=myfastinner2(Xdata,Y, sqrtminv, myalphaold,tildealpha, weight);

myhatalpha=cell(1,D);

%use edge and weight to obtain a weighted graph
%find the laplacian
%form eta||LB|_2 + (1-eta)||B||_2

if(strcmp(withCov,'True'))
   edges{1,D}=[]; %no edge information
   %vweight{1,D}=[];
   Tau(D)=0.00001;
end
%myL=cell(D,1);
for d=1:D
    p=size(Xdata{1,d},2);
    
    %if edge information is empty, then no group information
    %utilizes sida;
    
    if(~isempty(edges{1,d}))
%        edgesd= edges{1,d};
%        vweightd=vweight{1,d};
%        if(~isempty(vweightd))
%            %laplacian of weighted graph
%         WeightM=zeros(p,p); %for weight matrix
%         for j=1:length(edgesd)
%             indI=edgesd(j,1);
%             indJ=edgesd(j,2);
%             WeightM(indI,indJ)=vweightd(j);
%             WeightM(indJ,indI)=vweightd(j);
%             
%         end
%         Dv=sum(WeightM,2);
%         L=sparse(diag(Dv)-WeightM);
%         notZero=Dv~=0;
%         Dv2=zeros(length(Dv),1);
%         Dv2(notZero)=(Dv(Dv~=0)).^(-0.5);
%         
%         Dv=diag(Dv2);
%         nL=sparse(Dv*full(L)*Dv); %normalized Laplacian of weighted graph
%         myL{d}=sparse(nL);
%        elseif(isempty(vweightd)) %unweighted graph
%          AdjM=zeros(p,p);
%         for j=1:length(edgesd)
%             indI=edgesd(j,1);
%             indJ=edgesd(j,2);
%             AdjM(indI,indJ)=1;
%             AdjM(indJ,indI)=1;            
%         end  
%          Dv=sum(abs(AdjM),2);
%          L=diag(Dv)-AdjM;
%          notZero=Dv~=0;
%          Dv2=zeros(length(Dv),1);
%          Dv2(notZero)=(Dv(Dv~=0)).^(-0.5);
%         
%         Dv=diag(Dv2);
%         nL=sparse(Dv*full(L)*Dv); %normalized Laplacian of unweighted graph
%         myL{d}=sparse(nL);
%       end      
     
    nL=myL{d};
    %solve for alpha when network information is available
    cvx_begin quiet
        variable alphai(p,nK)
        mynorm2=sum(norms(alphai,2,2));
        LB=sparse(nL)*alphai;
        minimize( eta*sum(norms(LB,2,2)) +  (1-eta)*mynorm2)
     % minimize( eta*norm(LB) +  (1-eta)*mynorm2)  
      subject to
          norm(SepAndAssocd{1,d} - alphai*diag(tildelambda(:,d)), Inf)<=Tau(d);
       % end
    cvx_end
    
    elseif(isempty((edges{1,d})))       
       %solve for alpha when network information is not available
    %myL{d}=sparse(eye(p));   
    cvx_begin quiet
        variable alphai(p,nK)
        
         minimize(sum(norms(alphai,2,2)))
         subject to
          norm(SepAndAssocd{1,d} - alphai*diag(tildelambda(:,d)), Inf)<=Tau(d);
    cvx_end
    end
    
    
     %normalize alpha
        alphai(abs(alphai)<=10^-5)=0;
        if (min(sum(abs(alphai),1))==0)
            myalpha=alphai;
        else
        hatalpha=alphai./repmat(sqrt(sum(alphai.*alphai)),p,1);
        [Q,~]=qr(hatalpha);
        myalpha=Q(:,1:nK);
        myalpha(abs(myalpha)<=10^-5)=0;
        end
        myhatalpha{1,d}=myalpha;
end

clearvars alphai 