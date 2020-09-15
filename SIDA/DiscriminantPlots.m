%Sandra Safo
function plotout=DiscriminantPlots(Xtestdata,Ytest,hatalpha)
%Xtestdata is 1 by d cell array
% hatalpha is 1 by d cell array
% Ytest is n by 1 with classes labeled 1,2,...

dsizes = cellfun('size',Xtestdata,2);
D=length(dsizes);

if(length(unique(Ytest))==2)
    for d=1:D   
        ss2=Xtestdata{1,d}*hatalpha{1,d};
        ss21=ss2(Ytest==1);
        ss22=ss2(Ytest==2);
        xmin=min(ss21);
        xmax=max(ss21);
        xgrid=linspace(xmin,xmax,sum(Ytest==1));
        [ff1,xi1]=ksdensity(ss21, xgrid);

        xmin=min(ss22);
        xmax=max(ss22);
        xgrid=linspace(xmin,xmax,sum(Ytest==2));
        [ff2,xi2]=ksdensity(ss22, xgrid);

        plotout=figure();
        plot(ss21,ff1,'ro','linewidth',2,'MarkerSize',8); hold on;
        plot(ss22,ff2,'bv','linewidth',2,'MarkerSize',8);
        set(gca, 'fontsize',20);
        set(gca, 'linewidth',20);
        plot(xi1,ff1,'r','linewidth',2);
        plot(xi2,ff2,'b','linewidth',2);
        xlabel(['Discriminant Scores for X^' num2str(d)], 'fontsize',14)
        box on;
        set(gca,'LineWidth',2)
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        set(gca,'xticklabel',[])
        hold off;
        legend('Location','northeastoutside')
        title(['Discriminant plot for X^' num2str(d)],'fontsize',12)

    end
elseif(length(unique(Ytest))>2)
    for d=1:D
        Scores=Xtestdata{1,d}*hatalpha{1,d};
        plotout=figure();
        hold on;
        h1 = gscatter(Scores(:,1),Scores(:,2),Ytest,'krbmg','ov^d*',3);
        set(h1,'LineWidth',6)
        set(gca,'Fontsize',14);
        box on
        set(gca,'Linewidth',1.5);
        xlabel(['First Discriminant Scores for X^' num2str(d)], 'fontsize',14)
        ylabel(['Second Discriminant Scores for X^' num2str(d)], 'fontsize',14)
        hold off; 
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        legend('Location','northeastoutside')
        title(['Discriminant plot for X^' num2str(d)],'fontsize',12)

    end



end



