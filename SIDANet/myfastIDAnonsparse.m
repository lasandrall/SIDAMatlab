function[tildealphamat,tildelambda, myalphaoldmat,sqrtminvmat]=myfastIDAnonsparse(Xdata, Y,weight);

    %--------------------------------------------------------------------------
    %myfastIDAnonsparse.m: function to obtain nonsparse solution to integrative lda problem
    %and to obtain matrix needed in constraints
    %--------------------------------------------------------------------------


    %Input
    %Xdata:     1 by d cell array with each cell containing the n by p_d dataset
    %Y is a n x 1 vector;
    %nK is number of canonical discriminant vectors;
    %weight: balances separation and association. Default is 0.5.


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

    class=unique(Y);
    nc=length(class);
    
    %define weights
    w1=weight;
    w2=2*(1-weight)/(D*(D-1));

    %for storing matrices
    myalphaoldmat=cell(1,D);
    tildealphamat=cell(1,D);
    Crxd=cell(D,1);
    Sbx=cell(D,1);
    sqrtminvmat=cell(D,1);
    tildelambda=NaN(nc-1,D);
    rmyalphaoldmat=cell(1,D);
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
        Sbrx=Sbrx+ Sbrx';
        Sbx{d,1}=Sbrx ;

        lambda=sqrt(log(p)/n);
        if(n<p)
        Strx = Swrx+lambda*eye(n);
        Strx=Strx+ Strx';
        else
        Strx=Swrx +  Swrx';
        end
        %store between and within covariances
        Crxd{d,1}=Crx;

        %Set mybetaold and myalphaold as LDA solutions

        sqrtminv= mysqrtinv(Strx);
        sqrtminvmat{d,1}=sqrtminv;

        [myalphaold, tildelambdax1]=eigs(sqrtminv*Sbrx*sqrtminv,nc-1);
        [~,idx]=sort(real(diag(tildelambdax1)),'descend');
        myalphaold=real(myalphaold(:,idx'));
        myalphaold1= Ux1*myalphaold;
        myalphaoldmat{1,d}=myalphaold1./repmat(sqrt(sum(myalphaold1.*myalphaold1)),size(myalphaold1,1),1);
        rmyalphaoldmat{1,d}=myalphaold;
    end
       %Solution to integrative LDA


       %obtain association matrices
    for d=1:D
        dd=setdiff(1:D,d);
        %cross-covariance
        rSumassociation=0;
        for j=dd
        myalphaold=rmyalphaoldmat{1,j};
        Sdj=Crxd{d,1}*Crxd{j,1}'/(n-1);
        rassociation=(Sdj*myalphaold*myalphaold'*Sdj');
        rSumassociation=rSumassociation + rassociation + rassociation';
        %  association2{1,j}=association;
        end
        %solution to integrative LDA
        [tildealpha, tildelambdax]=eigs( sqrtminvmat{d,1}*( w1*Sbx{d,1} +  w2*rSumassociation)*sqrtminvmat{d,1},nc-1);
        [tildelambdax,idx]=sort(real(diag(tildelambdax)),'descend');
        tildealpha=tildealpha./repmat(sqrt(sum(tildealpha.*tildealpha)),size(tildealpha,1),1);
        tildealphamat{1,d}=real(tildealpha(:,idx'));
        tildelambda(:,d)=tildelambdax';
    end    


 
end

%Sandra E. Safo
%All rights reserved    