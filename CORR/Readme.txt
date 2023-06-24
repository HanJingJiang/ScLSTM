%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Usage of the source codes

The source codes are designed for clustering in single cell RNA-seq data.

There are two demo files: demo_corr.m' and demo_corrL.m
In order to use them, please just type demo_corr.m or demo_corrL.m in command window.

There are 5 data sets for testing purpose.
And the data sets with references are listed in the following.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Datasets References
1.Islet Data

Li, J., Klughammer, J., Farlik, M., Penz, T. et al. (2016) Single-cell transcriptomes
reveal characteristic features of human pancreatic islet cell types. EMBO Reports,
17,178-87.


2. Human Cancer Data

Ramskold, D.? Luo S., Wang, Y.C., et al. (2012) Full-length mRNA-Seq from singlecell
levels of RNA and individual circulating tumor cells. Nature Biotechnology, 30,
777-82.

3.Human Embryo Data

Yan, L., Yang, M., Guo, H. et al. (2013) Single-cell RNA-Seq profiling of human
preimplantation embryos and embryonic stem cells, Nature Structural and Molecular
Biology, 20, 1131-1139.

4.Mouse Data

Biase, F. H., Cao, X., & Zhong, S. (2014). Cell fate inclination within 2-cell and 4-cell mouse embryos revealed by single-cell rna sequencing. Genome Research, 24 (11), 1787.


5. Allodiploid Data

Li, X., Cui, X.L., Wang, J.Q., et al. (2016) Generation and Application of Mouse-Rat
Allodiploid Embryonic Stem Cells, Cell, 164, 279-92.




data=importdata('logdata//logbiase.csv');
label=importdata('label//biase.csv');
optnumber=Number_Corr(data);
%optnumber=max(label)-min(label)+1;
clusters = Corr_Clustering(data,optnumber,label);
dlmwrite('biase.txt',clusters,'delimiter','\t','newline','pc');


