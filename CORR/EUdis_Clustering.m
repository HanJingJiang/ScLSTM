function  [clusters]=EUdis_Clustering(A,optnumber,Labels)
%% Clustering Analysis
N = size(A,1); 
D = zeros(N);
for i = 1:N
    for j = 1:N
        c = (A(i,:)-A(j,:)).^2;
        D(i,j) = sqrt(sum(c(:)));
    end
end
D = D/max(max(D));
Dist = exp(-D) - eye(N);
clustTreeEuc = linkage(Dist,'weighted');        
clusters = cluster(clustTreeEuc,'MaxClust',optnumber);
%[h0,nodes0] = dendrogram(clustTreeEuc,0,'labels',Labels,'ColorThreshold','default');
%set(h0,'LineWidth',0.4)
%rotateticklabel(gca,90)




 
