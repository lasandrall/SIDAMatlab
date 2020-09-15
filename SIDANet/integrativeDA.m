  %gives solution to Theorem 1
function[mybeta,myLambda]=integrativeDA(Sigmaw,Sdj,Sbd)

%Sdj is a D by D cell, with diagonal covariance matrix, and off diagonals
%correlation matrix;
%Sigmaw- D by 1 cell within class matrix for each view
%Sbd- D by 1 cell between class matrix for each view

rng('default');

thresh=10^-5;
maxiteration=200;
nK=rank(Sbd{1,1});

D=length(Sigmaw);
for d=1:D
    d1=size(Sigmaw{d,1},1);
    BetaOld=1*randn(d1,nK);
    BetaOld=BetaOld./repmat(sqrt(sum(BetaOld.*BetaOld)),d1,1);
    mybetaold2{1,d}=BetaOld;
end


mybetaold=mybetaold2;
mybeta=[];
for iter=1:maxiteration
    for d=1:D

        dd=setdiff(1:D,d);
        sigma_wd=Sigmaw{d,1};
        Sb_d=Sbd{d,1};
        Sw=sigma_wd + sigma_wd';
        Sb=Sb_d + Sb_d';

        sumassocd=0;

        for j=dd;
            betaold=mybetaold{1,j};
            Sdj2=Sdj{d,j};
            assocd=Sdj2*betaold*betaold'*Sdj2';
            sumassocd=sumassocd + assocd + assocd';
        end

        %eigenvalue decomposition
        SepAssoc=(Sw)\(Sb + sumassocd);
        [Ud,Dd]=eigs(SepAssoc,nK);
         Ud=Ud./repmat(sqrt(sum(Ud.*Ud)),size(Ud,1),1);
        [Dd,idx]=sort(real(diag(Dd)),'descend');
         mybeta{1,d}=real(Ud(:,idx'));
         myLambda{d,1}=Dd;
    end

        mydiff=cellfun( @(xold,xnew) abs(xold)- abs(xnew), mybetaold, mybeta,'UniformOutput',false); %use UniformOutput false if want to reterun vectors/matrices
        diffalpha=cellfun( @(x) norm(x,inf),mydiff,'UniformOutput',false); %infinite norm of each direction vector for each D
        
               if( max(cell2mat(diffalpha))< thresh)
                   break
               else
                   mybetaold=mybeta;
  
               end
end



