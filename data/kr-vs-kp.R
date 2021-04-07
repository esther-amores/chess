#DATASET:
#https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29

#VARIABLE DESCRIPTION:
#http://notolog.blogspot.com/2011/01/features-of-uci-chess-data-sets.html

library(bnlearn)
library(gRain)
library(rchess)
library(Rgraphviz)
library(tidyverse)
library(MASS) # per realitzar el stepwise
library(readr)
library(xtable)

chss <- Chess$new()
plot(chss)

chss2 <- Chess$new("8/P7/8/1k6/5r2/8/K7/8 w - - 0 1")
plot(chss2)

#### Importar les dades: ####
dades <- read_csv("kr-vs-kp.data",
                  col_names = c("bkblk","bknwy","bkon8","bkona","bkspr",
                                "bkxbq","bkxcr","bkxwp","blxwp","bxqsq",
                                "cntxt","dsopp","dwipd","hdchk","katri",
                                "mulch","qxmsq","r2ar8","reskd","reskr",
                                "rimmx","rkxwp","rxmsq","simpl","skach",
                                "skewr","skrxp","spcop","stlmt","thrsk",
                                "wkcti","wkna8","wknck","wkovl","wkpos",
                                "wtoeg", "target"))

anyNA(dades)

# Passar a factor totes les variables
dades <- lapply(dades, factor) %>%
  as.data.frame()

str(dades)
apply(dades, 2, table)

#### Preprocessing de les dades: ####
model <- glm(target ~ ., data = dades, family = "binomial")
xtable(model)

stepwise <- stepAIC(model, direction = "both")
# el millor model escollit després de fer el mètode Stepwise segons valor del AIC
# treu les variables: stlmt, reskd, skewr, bkspr, simpl, wtoeg, dwipd, spcop

# Les variables escollides les guardem en aquest nou objecte:
dades <- stepwise$model
dades <- dades[c(2:ncol(dades),1)]

# I com que farem split-validation per poder validar finalment el model, 
# dividim la base de dades, a l'atzar, en dos trossos,
# training (80%), amb el qual aprendrem la xarxa, i validate (20%).

# Farem servir una llavor seed=666, per a que la nostra recerca sigui
# reproduible:

splitdf <- function(dataframe, seed = NULL){
  if(!is.null(seed)) set.seed(seed)
  index <- 1:nrow(dataframe)
  trainingindex <- sample(index,length(index)*0.8)
  trainingset <- dataframe[trainingindex, ]
  validateset <- dataframe[-trainingindex, ]
  list(trainingset = trainingset, validateset = validateset)
}

splits <- splitdf(dades, seed = 666)
# ens dona una llista: dos data frames anomenats "training" i "validate"
# Nombre d'observacions en cada subconjunt de dades:
lapply(splits, nrow)

training <- splits$trainingset
test <- splits$validateset

# - Fem el k-fold cross validation:
kfold <- function(dades, k, seed = NULL){
  if(!is.null(seed)) set.seed(seed)
  trainingset <- list()
  testset <- list()
  
  for(i in 1:k){
    dades$id <- sample(1:k, nrow(dades), replace = TRUE)
    folds <- 1:k
    
    trainingset[[i]] <- subset(dades, id %in% folds[-i])
    testset[[i]] <- subset(dades, id %in% c(i))
  }
  
  return(list(training = trainingset, test = testset))
}

Bayes_clasificator <- function(dades, wl, bl, k, seed){
  #### 1er: Aprendre l'estructura de les dades (amb totes les variables) ####
  # Generem la llista amb tots els folds i treiem la columna dels id's
  # tant pels trainings com pels tests
  
  training <- kfold(dades, k, seed)$training %>%
    lapply(function(x) x[,-ncol(x)])
  
  test <- kfold(dades, k, seed)$test %>%
    lapply(function(x) x[,-ncol(x)])
  
  # - Aprendre l'estructura de les dades per tots els possibles folds
  xarxa <- lapply(training, function(x) hc(x, score = "bic", whitelist = wl, blacklist = bl))
  
  # Fem el plot per cada fold
  lapply(xarxa, graphviz.plot)
  
  #### 2on: Estimem els paràmetres de la xarxa, pel mètode MLE ####
  xarxa.estimada <- list()
  for(i in 1:k)
    xarxa.estimada[[i]] <- bn.fit(xarxa[[i]], training [[i]], method = "mle")
  
  #### 3er: Fem la validació, estimar la classe per tots els conjunts tests ####
  # passem la xarxa a format gRain
  xarxa.grain <- lapply(xarxa.estimada, function(x) suppressWarnings(as.grain(x)))
  
  distribucio <- NULL
  prediccio <- NULL
  CL <- NULL
  
  for(i in 1:k){
    distribucio[[i]] <- list()
    prediccio[[i]] <- list()
    CL[[i]] <- list()
    for(j in 1:nrow(test[[i]])){
      if(is.numeric(predict(xarxa.grain[[i]], 
                            response="target", 
                            test[[i]][j,], 
                            predictors=atributes, 
                            type="dist")$pred[[1]][1,1])==FALSE)
      {
        prediccio[[i]][[j]]<-NA
        CL[[i]][[j]]<-0
        distribucio[[i]][[j]]<-c(rep(0,2))
      }
      else
      {
        distribucio[[i]][[j]] <- list(predict(xarxa.grain[[i]],
                                              response="target",
                                              test[[i]][j,],
                                              predictors=atributes,
                                              type="dist")$pred[[1]][1,])
        prediccio[[i]][[j]] <- names(distribucio[[i]][[j]][[1]])[which.max(distribucio[[i]][[j]][[1]])]
        CL[[i]][[j]]<-max(distribucio[[i]][[j]][[1]])
      }
    }
  }
  
  #### 4t: Càlcul de la matriu de confusió i les mesures d'avaluació ####
  matriu.confusio <- list()
  for(i in 1:k)
    matriu.confusio[[i]] <- as.matrix(table(unlist(prediccio[[i]]), test[[i]]$target))
  
  # Accuracy
  accuracy <- lapply(matriu.confusio, function(x) round((sum(x[1,1] + x[2,2])/sum(x))*100,2))
  accuracy <- unlist(accuracy)
  
  # True Positive Rate
  TPR <- lapply(matriu.confusio, function(x) round((sum(x[1,1])/sum(x[1,1] + x[2,1]))*100,2))
  TPR <- unlist(TPR)
  
  # True Negative Rate
  TNR <- lapply(matriu.confusio, function(x) round((sum(x[2,2])/sum(x[2,2] + x[1,2]))*100,2))
  TNR <- unlist(TNR)
  
  # Balanced Accuracy
  baccuracy <- round((TPR+TNR)/2,2)
  
  # Positive Predictive Value
  PPV <- lapply(matriu.confusio, function(x) round((sum(x[1,1])/sum(x[1,1] + x[1,2]))*100,2))
  PPV <- unlist(PPV)
  
  # F1 score
  F1 <- round(2*((PPV*TPR)/(PPV+TPR)),2)
  
  Mesures <- rbind(accuracy, TPR, TNR, 
                   baccuracy, PPV, F1)
  Mesures <- apply(Mesures, 1, mean)
  Mesures <- matrix(Mesures, ncol = 1)
  rownames(Mesures) <- c("Accuracy","True Positive Rate", 
                         "True Negative Rate", "Balanced Accuracy", 
                         "Positive Predictive Value", "F1 score")
  colnames(Mesures) <- "Mètriques"
  return(list(Confusion_matrix = matriu.confusio, Metrics = Mesures))
}

####################################
#      Naive Bayes classifier      #
####################################

atributes <- colnames(dades[-ncol(dades)])
wl <- data.frame(from = rep("target", length(atributes)), to = atributes)
bl <- rbind(
  expand.grid(atributes, atributes),
  data.frame(Var1 = atributes, Var2= rep("target", length(atributes))),
  data.frame(Var1 = "target", Var2= "target")
)

Naive <- Bayes_clasificator(training, wl, bl,5, 666)

xtable(Naive$Metrics)

####################################
#    Augmented Bayes classifier    #
####################################
atributes <- colnames(dades[-ncol(dades)])
wl <- data.frame(from = rep("target", length(atributes)), to = atributes)

Augmented <- Bayes_clasificator(training, wl, bl = NULL, 5, 666)

Augmented$Metrics

xtable(Naive$Metrics)

###################################
#  Tria del clasificador bayesià  #
###################################

Mesures <- cbind(Naive$Metrics, Augmented$Metrics)
colnames(Mesures) <- c("Naive Bayes", "Augmented Naive Bayes")

xtable(Mesures)

#Veiem que de totes les mesures que hem calculat per saber com de bé estima 
#les xarxes bayesianes, les del "Augmented Naive Bayes" són més elevades (>95%) 
#que les del "Naive Bayes" y per tant triem el model de Augmented Naive Bayes per 
#per modelar de nou pero ara amb tota la base de dades training sense fer k-fold

####################################
# Augmented Bayes classifier split #
####################################
#### 1er: Aprendre l'estructura de les dades (amb totes les variables) ####

# - Introduim la white list:
atributes <- colnames(dades[-ncol(dades)])

wl <- data.frame(from = rep("target", length(atributes)), to = atributes)

xarxa <- hc(training, score = "bic", whitelist = wl)

graphviz.plot(xarxa)

xarxa.estimada <- bn.fit(xarxa, training, method = "mle")
xarxa.grain <- suppressWarnings(as.grain(xarxa.estimada))

prediccio <- NULL
CL <- NULL
prueba <- NULL
distribucio <- NULL

for (j in 1:nrow(test)){
  if (is.numeric(predict(xarxa.grain, response="target", 
                         test[j,], predictors = atributes,
                         type = "dist")$pred[[1]][1,1])==FALSE){
    prediccio[[j]] <- NA
    CL[[j]] <- 0
    distribucio[[j]] <- c(rep(0,2))
  }
  else{
    prueba[[j]] <- predict(xarxa.grain, response="target", 
                           test[j,], predictors = atributes, 
                           type="dist")
    distribucio[[j]] <- prueba[[j]]$pred[[1]]
    prediccio[[j]] <- dimnames(distribucio[[j]])[[2]][which.max(distribucio[[j]])]
    CL[[j]] <- max(distribucio[[j]])
  }
}

matriu.confusio <- as.matrix(table(prediccio, test$target))
matriu.confusio

# Accuracy
Accuracy_final <- round((sum(matriu.confusio[1,1] + 
                               matriu.confusio[2,2])/
                                  sum(matriu.confusio))*100,2)

# True Positive Rate
TPR_final <- round((sum(matriu.confusio[1,1])/
                      sum(matriu.confusio[1,1] + 
                            matriu.confusio[2,1]))*100,2)

# True Negative Rate
TNR_final <- round((sum(matriu.confusio[2,2])/
                      sum(matriu.confusio[2,2] + 
                            matriu.confusio[1,2]))*100,2)

# Balanced Accuracy
baccuracy_final <- round((TPR_final+TNR_final)/2,2)

# Positive Predictive Value
PPV_final <- round((sum(matriu.confusio[1,1])/
                      sum(matriu.confusio[1,1] + 
                            matriu.confusio[1,2]))*100,2)

# F1 score
F1_final <- round(2*((PPV_final*TPR_final)/(PPV_final+TPR_final)),2)

Mesures <- rbind(Accuracy_final, TPR_final, TNR_final, 
                 baccuracy_final, PPV_final, F1_final)
rownames(Mesures) <- c("Accuracy","True Positive Rate", 
                       "True Negative Rate", "Balanced Accuracy", 
                       "Positive Predictive Value", "F1 score")
colnames(Mesures) <- "Métriques"
Mesures

xtable(Mesueres)

# Observem finalment que la xarxa bayesiana Augmented Naive Bayes obté unes 
# métriques molt bones (>94%) 

####################################
# Augmented Bayes classifier final #
####################################
#### 1er: Aprendre l'estructura de les dades (amb totes les variables) ####

# - Introduim la white list:
atributes <- colnames(dades[-ncol(dades)])

wl <- data.frame(from = rep("target", length(atributes)), to = atributes)

xarxa <- hc(dades, score = "bic", whitelist = wl)

graphviz.plot(xarxa)

xarxa.estimada <- bn.fit(xarxa, dades, method = "mle")
xarxa.grain <- suppressWarnings(as.grain(xarxa.estimada))


##########################################################################
# Ara podem simular noves partides d'escacs on es doni aquesta situació de 
# final de partida onles peces blanques tenen el Rei i un Peó mentres que les 
# negres tenen el Rei i un torre.

chss3 <- Chess$new("8/P7/8/2k5/2r5/8/K7/8 w - - 0 1")
plot(chss3)

a <- lapply(dades, function(x) sample(levels(x),1))
a <- unlist(a[-c(length(a))])

a <- dades[sample(1:nrow(dades),1),]
a <- a[-c(length(a))]

a <- c(TRUE,FALSE,FALSE,FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,
       "g",FALSE,"n",TRUE,FALSE,FALSE,FALSE,TRUE,FALSE,FALSE,FALSE,TRUE,TRUE,
       TRUE,TRUE,FALSE,FALSE,TRUE,TRUE,FALSE,FALSE,TRUE,TRUE,"n")

xarxa.evid <- setEvidence(xarxa.grain, nodes = atributes,
                          states = a)
evid <- xarxa.evid$evidence
qq <- querygrain(xarxa.evid, nodes = c("target"), type = "marginal")
distribucio <- qq$target
prediccio <- dimnames(distribucio)[[1]][which.max(distribucio)]
CL <- round(100*max(distribucio), 2)
reincidence.risc <- round(100*distribucio[2], 2)

Resultat <- matrix(c(prediccio, CL,reincidence.risc),nrow=1)
Resultat <- as.data.frame(Resultat)
colnames(Resultat) <- c("Predicció","Confidence Level (CL) en %",
                        "Risc de reincidència")
evid
Resultat
