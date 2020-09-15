%perform classificaton using nearest centroid;
function [Predclass] = sida_classify(Projtest, Projtrain, Y,classMethod); 

%Projtest is projection of testing data onto estimated basis direction vectors
%Projtrain is projection of training data onto estimated basis direction vectors

class=unique(Y);
nc=length(class);
ntest=size(Projtest{1,1},1);
if(strcmp(classMethod,'Separate'))
         %nearest centroid for X
        D=size(Projtest,2);
        for d=1:D
            ProjtestXd=Projtest{1,d};
            ProjtrainXd=Projtrain{1,d};
            mProjtrain=grpstats(ProjtrainXd,Y);
            Projmv=[(1:nc)' mProjtrain];
            distv=[];
            for j=1:nc
                which=Projmv(:,1)==j;
                rProjm=repmat(Projmv(which,2:end),ntest,1);
                %euclidean distance 
                sqdiff=(ProjtestXd-rProjm).^2;
                dist1=(sum(sqdiff,2)).^0.5;
                jrep=j*ones(ntest,1);
                distv=[distv;[jrep dist1]];
            end

        %The following code outputs the assigned class
        rdistvX= reshape(distv(:,2), ntest, nc);
        [~, predclassX]=min(rdistvX(:,1:nc),[],2);
        PredclassSeparate{1,d}=predclassX;
        end
        Predclass=cell2mat(PredclassSeparate);

        elseif(strcmp(classMethod,'Joint'))
        %classification for joint
        ProjtestJoint=cell2mat(Projtest);
        ProjtrainJoint=cell2mat(Projtrain);
        ntest=size(ProjtestJoint,1);
        mProjtrain=grpstats(ProjtrainJoint,Y);
        Projmv=[(1:nc)' mProjtrain];
        distv=[];
        for j=1:nc
            which=Projmv(:,1)==j;
            rProjm=repmat(Projmv(which,2:end),ntest,1);
            %euclidean distance 
            sqdiff=(ProjtestJoint-rProjm).^2;
            dist1=(sum(sqdiff,2)).^0.5;
            jrep=j*ones(ntest,1);
            distv=[distv;[jrep dist1]];
        end
        %The following code outputs the assigned class
        rdistvJoint= reshape(distv(:,2), ntest, nc);
        [~, Predclass]=min(rdistvJoint(:,1:nc),[],2);
end


% if(strcmp(classMethod,'randomAssign'))
% %if disagreement, random assignment
% checkAgreement=[predclassX predclassZ];
% checkAgreement2=[checkAgreement checkAgreement(:,1)];
% checkAgreement2(checkAgreement(:,1)~=checkAgreement(:,2),3)=unidrnd(nc,1,1);
% predclass=checkAgreement2(:,3);
% elseif(strcmp(classMethod,'minboth'))
% %if disagreement, random assignment
% %find where there is disagreement 
% rdistv=[1:nc 1:nc ; rdistvX rdistvZ];
% rdistv2=NaN(size(rdistv,1),nc);
% for j=1:nc;
%     which=rdistv(1,:)==j;
%     rdistv2(:,j)=min(rdistv(:,which),[],2);
% end
% [mini, predclass]=min(rdistv2(2:end,1:nc),[],2);

end