function myL =myNLaplcianG(Xdata,edges,vweight)

[a,b]=size(Xdata);
D=max(a,b);
if(a>b)
  Xdata=Xdata';
end
for d=1:D
    p=size(Xdata{1,d},2);
    
    %if edge information is empty, then no graph information
    %utilizes sida;
    
    if(~isempty(edges{1,d}))
       edgesd= edges{1,d};
       vweightd=vweight{1,d};
       if(~isempty(vweightd))
           %laplacian of weighted graph
        WeightM=zeros(p,p); %for weight matrix
        for j=1:length(edgesd)
            indI=edgesd(j,1);
            indJ=edgesd(j,2);
            WeightM(indI,indJ)=vweightd(j);
            WeightM(indJ,indI)=vweightd(j);
            
        end
        Dv=sum(WeightM,2);
        L=sparse(diag(Dv)-WeightM);
        notZero=Dv~=0;
        Dv2=zeros(length(Dv),1);
        Dv2(notZero)=(Dv(Dv~=0)).^(-0.5);
        
        Dv=diag(Dv2);
        nL=sparse(Dv*full(L)*Dv); %normalized Laplacian of weighted graph
        myL{d}=sparse(nL);
       elseif(isempty(vweightd)) %unweighted graph
         AdjM=zeros(p,p);
        for j=1:length(edgesd)
            indI=edgesd(j,1);
            indJ=edgesd(j,2);
            AdjM(indI,indJ)=1;
            AdjM(indJ,indI)=1;            
        end  
         Dv=sum(abs(AdjM),2);
         L=diag(Dv)-AdjM;
         notZero=Dv~=0;
         Dv2=zeros(length(Dv),1);
         Dv2(notZero)=(Dv(Dv~=0)).^(-0.5);
        
        Dv=diag(Dv2);
        nL=sparse(Dv*full(L)*Dv); %normalized Laplacian of unweighted graph
        myL{d}=sparse(nL);
       end 
       
     elseif(isempty((edges{1,d})))  
           myL{d}=sparse(eye(p));   
    end
end