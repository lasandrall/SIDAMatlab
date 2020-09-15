function[SepAndAssocd, Ux]=myfastinner2(Xdata,Y, sqrtminv, myalphaoldmat,tildealphamat,weight);

%--------------------------------------------------------------------------
%myfastnonsparse.m: function to obtain nonsparse solution to lda problem
%and to obtain matrix needed in constraints
%--------------------------------------------------------------------------


%Input
%X1 is a n x p dataset
%X2 is a n x q dataset 
%Y is a n x 1 vector;
%nK is number of canonical discriminant vectors;
%nonsparse is structure of covariance matrices for X and Y. Either Iden or
%Ridge   

%check input;
[a,b]=size(Y);
if(a>b)
    Y=Y';
end
[a,b]=size(Xdata);
D=max(a,b);
if(a>b)
  Xdata=Xdata';
end

%define weights
w1=weight;
w2=2*(1-weight)/(D*(D-1));

%for storing matrices
separationd=cell(1,D);
associationd=cell(1,D);
SepAndAssocd=cell(1,D);
for d=1:D
    
    %obtain between and within covariances for each dataset
    myX=Xdata{1,d};
    myX=myX';
    [p,n]=size(myX);

    %orginal LDA solution inv(St)Sb for first data
    [Ux1, W, V] = svd(myX, 'econ');
    %Uxd{d,1}=Ux1;
    R = W*V';
    
    rdata=[Y;R];
    mrd=grpstats(rdata',Y);
    mr=mean(rdata(2:end,:),2);
    
    class=unique(Y);
    nc=length(class);
    C=cell(1,nc);
    for i=class;
    which=rdata(1,:)==i;
    which2=mrd(:,1)==i;
    C{1,i}=rdata(2:end,which)-repmat(mrd(which2,2:end)', 1,sum(rdata(1,:)==i));
    end;     
    C=cell2mat(C);

    Crx = R - repmat(mr,1,n);
    Srx=(Crx*Crx')/(n-1);
    Swrx=(C*C')/(n-1);
    Sbrx=Srx-Swrx;
    Sbrx=Sbrx+Sbrx';

    lambda=sqrt(log(p)/n);
    if(n<p)
    Strx = Swrx+lambda*eye(n);
    Strx=Strx+Strx';
    else
    Strx=Swrx + Swrx';
    end
    %store between and within covariances
    Swx{d,1}=Strx;
    Sbx{d,1}=Sbrx;
    Ux{d,1}=Ux1;
    Crxd{d,1}=Crx;
    sqrtminvStrx=sqrtminv{d,1};
    separationd{1,d}=w1*Ux1*sqrtminvStrx*Sbrx*sqrtminvStrx*tildealphamat{1,d};
    
end

%obtain association matrices
for d=1:D
    dd=setdiff(1:D,d);
    %cross-covariance
    Sumassociation=0;
    for j=dd
    myalphaold=myalphaoldmat{1,j};  
    Sdj=Crxd{d,1}*Crxd{j,1}';
    assoc2=(Sdj*Ux{j,1}'*(myalphaold*myalphaold')*Ux{j,1}*Sdj');
    association=Ux{d,1}*sqrtminv{d,1}*(assoc2+assoc2')*Ux{d,1}'*Ux{d,1}*sqrtminv{d,1}*tildealphamat{1,d}/(n-1)^2;
    Sumassociation=Sumassociation + association ;
    %  association2{1,j}=association;
    end
    associationd{1,d}=w2*Sumassociation;
    SepAndAssoc=separationd{1,d}+ Sumassociation;
    SepAndAssoc=SepAndAssoc./repmat(norm(SepAndAssoc,inf),1,size(SepAndAssoc,2));
    SepAndAssocd{1,d}= SepAndAssoc;
    %SepAndAssoc./repmat(sqrt(sum(SepAndAssoc.*SepAndAssoc)),length(SepAndAssoc),1);
    
end    
       
%Sandra E. Safo
%All rights reserved    