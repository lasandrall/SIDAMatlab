%Sandra Safo
function plotout=CorrelationPlots(Xtestdata,Ytest,hatalpha)
%Xtestdata is 1 by d cell array
% hatalpha is 1 by d cell array
% Ytest is n by 1 with two classes labeled 1 and 2

dsizes = cellfun('size',Xtestdata,2);
D=length(dsizes);
mycomb=combnk(1:D,2);
    for d=1:size(mycomb,1)
        dd=mycomb(d,:);
        Scoresd=Xtestdata{1,dd(1,1)}*hatalpha{1,dd(1,1)};   
        plotout=figure();
        Scoresj=Xtestdata{1,dd(1,2)}*hatalpha{1,dd(1,2)};   
        
        %calculate RV coefficient
        Xdd=Xtestdata{1,dd(1,1)}*hatalpha{1,dd(1,1)};
        Xjj=Xtestdata{1,dd(1,2)}*hatalpha{1,dd(1,2)};
        Sdd= cov(Xdd);
        Sjj=cov(Xjj);
        %center matrices
        cSd=Xdd-repmat(mean(Xdd),size(Xdd,1),1);
        cSj=Xjj-repmat(mean(Xjj),size(Xjj,1),1);
        Sdj=cSd'*cSj/(size(cSd,1)-1);
        RVCoeff=trace(Sdj*Sdj')/(trace(Sdd*Sdd')^.5*trace(Sjj*Sjj')^0.5);
        RVCoeff=round(RVCoeff,2);
        hold on;
        h1 = gscatter(Scoresd(:,1),Scoresj(:,1),Ytest,'krbmg','ov^d*',3);
        set(h1,'LineWidth',6)
        set(gca,'Fontsize',14);
        box on
        set(gca,'Linewidth',1.5);
        xlabel(['First Discriminant Scores for X^' num2str(dd(1,1))], 'fontsize',14)
        ylabel(['First Discriminant Scores for X^' num2str(dd(1,2))], 'fontsize',14)
        hold off; 
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        legend('Location','northeastoutside')
        title(['Correlation plot for X^' num2str(dd(1,1)) 'and X^' num2str(dd(1,2)) ', \rho =' num2str(RVCoeff) ],'fontsize',12)

    end



end



