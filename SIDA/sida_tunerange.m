
function Tauvec= sida_tunerange(Xdata,Y,ngrid,standardize,weight,withCov)

 %--------------------------------------------------------------------------
%sida_tunerange.m: function to provide tuning parameter lower and upper bounds
%to estimate canonical discriminant vectors;
%--------------------------------------------------------------------------


%DATE: April 30, 2019
%
%MATLAB CODE WAS WRITTEN BY SANDRA E. SAFO (seaddosafo@gmail.com)
%Xdata is 1 by d cell array with each cell containing the n by p_d dataset
%Y is n by 1 vector of class membership;
%ngrid is number of grid points
%standardize- whether to standardize or not
%weight balances association and separation

%Output
%Tauvec is a 1 by 1 cell array containing a d x 1 cell array (each dth entry is a ngrid by 1 vector)
%of tuning parameters 

if strcmp(standardize,'True')
   Xdata=cellfun(@(x) mynormalize(x), Xdata,'UniformOutput',false);
end

%size of each data;
dsizes = cellfun('size',Xdata,2);
D=length(dsizes);
if(strcmp(withCov,'True'))
    D=D-1;
end
n = cellfun('size',Xdata,1);
n=n(1);
nK=length(unique(Y))-1;

%obtain  nonsparse input

    [tildealphamat,tildelambda, myalphaoldmat,sqrtminv]=myfastIDAnonsparse(Xdata, Y,weight);
    [SepAndAssoc,~]=myfastinner2(Xdata,Y,sqrtminv,myalphaoldmat,tildealphamat,weight);

    %obtain upper bound as inf norm of SepAndAssoc
    ubx=cellfun( @(x) norm(x,inf),SepAndAssoc);
    
    %tuning range
     TauX1range=[sqrt(log(dsizes(:,1:D))./(n)).*ubx(:,1:D) ; ubx(:,1:D)/1.2]';   
     cc=mat2cell(TauX1range,ones(D,1),[1,1]);
     Taugrid=cellfun(@(x1,x2) linspace(x1,x2,ngrid+1), cc(:,1),cc(:,2), 'UniformOutput',false);

    myperx= cellfun(@(x) prctile(x(:,1:ngrid), [10 15 20 25 35 45]),Taugrid,'UniformOutput',false);
    myperx=cell2mat(myperx);
   
    for loc=1:6
        Tau=myperx(:,loc);
        myalpha=sida_inner(Xdata,Y, sqrtminv,myalphaoldmat,tildealphamat, tildelambda, Tau,weight,withCov);
        nonzerosd=cell2mat(cellfun( @(myx) sum(myx~=0), myalpha,'UniformOutput',false));
        nonzerosd1=reshape(nonzerosd,nK,length(nonzerosd)/nK);
        myres=nonzerosd1./repmat(dsizes,nK,1);
        if(all(myres(:)<=0.25))
            break
        end
     
    end  

    %final grid
    Tau1=mat2cell(myperx(:,loc),ones(1,D),1);
    Tauvec1=cellfun(@(x1,x2) linspace(x1,x2,ngrid+1), Tau1,cc(:,2), 'UniformOutput',false);
    Tauvec{1,1}=cellfun(@(x) x(:,1:ngrid), Tauvec1, 'UniformOutput',false);
 
 %   Tauvec{1,1}=Taugrid;
    %All rights researved


