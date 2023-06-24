%%  biase
disp('biase')
data=importdata('logdata//logbiase.csv');
label=importdata('label//biase.csv');
[optnumber,dis]=Number_Corr(data);
clusters = Corr_Clustering(data,optnumber,label,dis);

dlmwrite('CORR/biase.txt',clusters,'delimiter','\t','newline','pc');
dlmwrite('CORR/biasenum.txt',optnumber,'delimiter','\t','newline','pc');

clusters1 = Spearman_Clustering(data,optnumber,label);
dlmwrite('SP/biase.txt',clusters1,'delimiter','\t','newline','pc');

clusters2 = EUdis_Clustering(data,optnumber,label);
dlmwrite('EU/biase.txt',clusters2,'delimiter','\t','newline','pc');

clusters3 = Cos_Clustering(data,optnumber,label);
dlmwrite('COS/biase.txt',clusters3,'delimiter','\t','newline','pc');
%% camp
clc
disp('camp')
data=importdata('logdata//logcamp.csv');
label=importdata('label//camp.csv');
[optnumber,dis]=Number_Corr(data);
clusters = Corr_Clustering(data,optnumber,label,dis);

dlmwrite('CORR/camp.txt',clusters,'delimiter','\t','newline','pc');
dlmwrite('CORR/campnum.txt',optnumber,'delimiter','\t','newline','pc');

clusters1 = Spearman_Clustering(data,optnumber,label);
dlmwrite('SP/camp.txt',clusters1,'delimiter','\t','newline','pc');

clusters2 = EUdis_Clustering(data,optnumber,label);
dlmwrite('EU/camp.txt',clusters2,'delimiter','\t','newline','pc');

clusters3 = Cos_Clustering(data,optnumber,label);
dlmwrite('COS/camp.txt',clusters3,'delimiter','\t','newline','pc');
%% deng
clc
disp('deng')
data=importdata('logdata//logdeng.csv');
label=importdata('label//deng.csv');
[optnumber,dis]=Number_Corr(data);
clusters = Corr_Clustering(data,optnumber,label,dis);

dlmwrite('CORR/deng.txt',clusters,'delimiter','\t','newline','pc');
dlmwrite('CORR/dengnum.txt',optnumber,'delimiter','\t','newline','pc');

clusters1 = Spearman_Clustering(data,optnumber,label);
dlmwrite('SP/deng.txt',clusters1,'delimiter','\t','newline','pc');

clusters2 = EUdis_Clustering(data,optnumber,label);
dlmwrite('EU/deng.txt',clusters2,'delimiter','\t','newline','pc');

clusters3 = Cos_Clustering(data,optnumber,label);
dlmwrite('COS/deng.txt',clusters3,'delimiter','\t','newline','pc');
%% goolam
clc
disp('goolam')
data=importdata('logdata//loggoolam.csv');
label=importdata('label//goolam.csv');
[optnumber,dis]=Number_Corr(data);
clusters = Corr_Clustering(data,optnumber,label,dis);

dlmwrite('CORR/goolam.txt',clusters,'delimiter','\t','newline','pc');
dlmwrite('CORR/goolamnum.txt',optnumber,'delimiter','\t','newline','pc');

clusters1 = Spearman_Clustering(data,optnumber,label);
dlmwrite('SP/goolam.txt',clusters1,'delimiter','\t','newline','pc');

clusters2 = EUdis_Clustering(data,optnumber,label);
dlmwrite('EU/goolam.txt',clusters2,'delimiter','\t','newline','pc');

clusters3 = Cos_Clustering(data,optnumber,label);
dlmwrite('COS/goolam.txt',clusters3,'delimiter','\t','newline','pc');
%% kolo
clc
disp('kolo')
data=importdata('logdata//logkolo.csv');
label=importdata('label//kolo.csv');
[optnumber,dis]=Number_Corr(data);
clusters = Corr_Clustering(data,optnumber,label,dis);

dlmwrite('CORR/kolo.txt',clusters,'delimiter','\t','newline','pc');
dlmwrite('CORR/kolonum.txt',optnumber,'delimiter','\t','newline','pc');

clusters1 = Spearman_Clustering(data,optnumber,label);
dlmwrite('SP/kolo.txt',clusters1,'delimiter','\t','newline','pc');

clusters2 = EUdis_Clustering(data,optnumber,label);
dlmwrite('EU/kolo.txt',clusters2,'delimiter','\t','newline','pc');

clusters3 = Cos_Clustering(data,optnumber,label);
dlmwrite('COS/kolo.txt',clusters3,'delimiter','\t','newline','pc');
%% li
clc
disp('li')
data=importdata('logdata//logli.csv');
label=importdata('label//li.csv');
[optnumber,dis]=Number_Corr(data);
clusters = Corr_Clustering(data,optnumber,label,dis);

dlmwrite('CORR/li.txt',clusters,'delimiter','\t','newline','pc');
dlmwrite('CORR/linum.txt',optnumber,'delimiter','\t','newline','pc');

clusters1 = Spearman_Clustering(data,optnumber,label);
dlmwrite('SP/li.txt',clusters1,'delimiter','\t','newline','pc');

clusters2 = EUdis_Clustering(data,optnumber,label);
dlmwrite('EU/li.txt',clusters2,'delimiter','\t','newline','pc');

clusters3 = Cos_Clustering(data,optnumber,label);
dlmwrite('COS/li.txt',clusters3,'delimiter','\t','newline','pc');
%% pollen
clc
disp('pollen')
data=importdata('logdata//logpollen.csv');
label=importdata('label//pollen.csv');
[optnumber,dis]=Number_Corr(data);
clusters = Corr_Clustering(data,optnumber,label,dis);

dlmwrite('CORR/pollen.txt',clusters,'delimiter','\t','newline','pc');
dlmwrite('CORR/pollennum.txt',optnumber,'delimiter','\t','newline','pc');

clusters1 = Spearman_Clustering(data,optnumber,label);
dlmwrite('SP/pollen.txt',clusters1,'delimiter','\t','newline','pc');

clusters2 = EUdis_Clustering(data,optnumber,label);
dlmwrite('EU/pollen.txt',clusters2,'delimiter','\t','newline','pc');

clusters3 = Cos_Clustering(data,optnumber,label);
dlmwrite('COS/pollen.txt',clusters3,'delimiter','\t','newline','pc');
%% usoskin
clc
disp('usoskin')
data=importdata('logdata//logusoskin.csv');
label=importdata('label//usoskin.csv');
[optnumber,dis]=Number_Corr(data);
clusters = Corr_Clustering(data,optnumber,label,dis);

dlmwrite('CORR/usoskin.txt',clusters,'delimiter','\t','newline','pc');
dlmwrite('CORR/usoskinnum.txt',optnumber,'delimiter','\t','newline','pc');

clusters1 = Spearman_Clustering(data,optnumber,label);
dlmwrite('SP/usoskin.txt',clusters1,'delimiter','\t','newline','pc');

clusters2 = EUdis_Clustering(data,optnumber,label);
dlmwrite('EU/usoskin.txt',clusters2,'delimiter','\t','newline','pc');

clusters3 = Cos_Clustering(data,optnumber,label);
dlmwrite('COS/usoskin.txt',clusters3,'delimiter','\t','newline','pc');
%% yan
clc
disp('yan')
data=importdata('logdata//logyan.csv');
label=importdata('label//yan.csv');
[optnumber,dis]=Number_Corr(data);
clusters = Corr_Clustering(data,optnumber,label,dis);

dlmwrite('CORR/yan.txt',clusters,'delimiter','\t','newline','pc');
dlmwrite('CORR/yannum.txt',optnumber,'delimiter','\t','newline','pc');

clusters1 = Spearman_Clustering(data,optnumber,label);
dlmwrite('SP/yan.txt',clusters1,'delimiter','\t','newline','pc');

clusters2 = EUdis_Clustering(data,optnumber,label);
dlmwrite('EU/yan.txt',clusters2,'delimiter','\t','newline','pc');

clusters3 = Cos_Clustering(data,optnumber,label);
dlmwrite('COS/yan.txt',clusters3,'delimiter','\t','newline','pc');