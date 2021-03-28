#DATASET:
#https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29

#VARIABLE DESCRIPTION:
#http://notolog.blogspot.com/2011/01/features-of-uci-chess-data-sets.html
library(readr)
kr_vs_kp <- read_csv("kr-vs-kp.data", 
                     col_names = c("bkblk","bknwy","bkon8","bkona","bkspr",
                                   "bkxbq","bkxcr","bkxwp","blxwp","bxqsq",
                                   "cntxt","dsopp","dwipd","hdchk","katri",
                                   "mulch","qxmsq","r2ar8","reskd","reskr",
                                   "rimmx","rkxwp","rxmsq","simpl","skach",
                                    "skewr","skrxp","spcop","stlmt","thrsk",
                                   "wkcti","wkna8","wknck","wkovl","wkpos",
                                   "wtoeg", "target"))

View(kr_vs_kp)
