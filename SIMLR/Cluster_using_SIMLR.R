library("Matrix")
library("parallel")
library("igraph")
library("grDevices")
# load SIMLR
source("E:/deepimpute/SC for sister/test_SIMLR/R/SIMLR.R")
source("E:/deepimpute/SC for sister/test_SIMLR/R/compute.multiple.kernel.R")
source("E:/deepimpute/SC for sister/test_SIMLR/R/network.diffusion.R")
source("E:/deepimpute/SC for sister/test_SIMLR/R/utils.simlr.R")
source("E:/deepimpute/SC for sister/test_SIMLR/R/tsne.R")

dyn.load("E:/deepimpute/SC for sister/test_SIMLR/R/projsplx_R.dll")
setwd("E:/deepimpute/SC for sister/test_SIMLR/R")
data <- read.csv("logdata/logbiase.csv", header = FALSE)
label <- read.csv("label/biase.csv", header = FALSE)
label <-as.numeric(unlist(label))  #想直接算NMI的必要步骤
data <- t(data)
result = SIMLR(X = data, c = max(label) - min(label) + 1)
write.table(result$S, file = "SIMLR-biase.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
y_p = result$y$cluster

write.table(y_p, file = "biase.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
