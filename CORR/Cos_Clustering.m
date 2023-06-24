function  [clusters]=Cos_Clustering(A,optnumber,Labels)
%% Clustering Analysis
Dist = squareform(1-pdist(A,'cosine')); %count cos S matrix
clustTreeEuc = linkage(Dist,'weighted');        
clusters = cluster(clustTreeEuc,'MaxClust',optnumber);
%[h0,nodes0] = dendrogram(clustTreeEuc,0,'labels',Labels,'ColorThreshold','default');
%set(h0,'LineWidth',0.4)
%rotateticklabel(gca,90)




 
