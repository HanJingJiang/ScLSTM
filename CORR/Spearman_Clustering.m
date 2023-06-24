function  [clusters]=Spearman_Clustering(A,optnumber,Labels)


N = size(A,1);           
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Clustering Analysis
r1=corr(A.',A.','type','spearman');
%r1=corr(A.',A.','type','pearson');  %pearson
for i = 1 : N
    r1(i,i)=0;
end
Dist=squareform(r1);
clustTreeEuc = linkage(Dist,'weighted');        
clusters = cluster(clustTreeEuc,'MaxClust',optnumber);
%[h0,nodes0] = dendrogram(clustTreeEuc,0,'labels',Labels,'ColorThreshold','default');
%set(h0,'LineWidth',0.4)
%rotateticklabel(gca,90)




 
