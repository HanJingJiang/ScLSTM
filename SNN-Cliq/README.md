## SNN-Cliq
This algorithm is used to do the identification of cell types from single-cell transcriptomes. The original input is a gene expression matrix. Algorithm output the clusters of cells. This source code is downloaded from the authors' website.

Original article is :

Xu C, Su Z. Identification of cell types from single-cell transcriptomes using a novel clustering method.[J]. Bioinformatics, 2015, 31(12):1974-80.



---

### To use:
you should set right path and run the following matlab and python command

### run these two command in matlab

first

```data=importdata('biaseimproved.csv')```

the expression file should  have the same format as cell * gene expression matrix

second

```SNN(data,'SNN_graph.txt',k);```

the parameter k should be a certain number of neighbor(eg:  5)

### then you can run the command in python

```python Cliq.py -i SNN_graph.txt -o Cluster_Result.txt ```

-i is the name of input file 

-o is the name of output file which saves the clustering result

see more help information you could run ```python Cliq.py -h ```

