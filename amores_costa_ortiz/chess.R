##############
## PACKAGES ##
##############

library(bnlearn)
library(gRain)
library(rchess)
library(Rgraphviz)
library(tidyverse)
library(MASS) 
library(readr)
library(xtable)
library(papeR)


#############
## DATASET ##
#############

# https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29

# VARIABLE DESCRIPTION:
# http://notolog.blogspot.com/2011/01/features-of-uci-chess-data-sets.html


####################################
#      Naive Bayes classifier      #
####################################
####################################
#    Augmented Bayes classifier    #
####################################
###################################
#  Tria del clasificador bayesià  #
###################################
####################################
# Augmented Bayes classifier split #
####################################
####################################
# Augmented Bayes classifier final #
####################################

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


latex.table.fac(dades[1:ceiling(ncol(dades)/2)], table="longtable")
latex.table.fac(dades[(ceiling(ncol(dades)/2)+1):ncol(dades)], table="longtable")


model <- glm(target ~ ., data = dades, family = "binomial")


stepwise <- stepAIC(model, direction = "both")
# el millor model escollit després de fer el mètode Stepwise segons valor del AIC
# treu les variables: stlmt, reskd, skewr, bkspr, simpl, wtoeg, dwipd, spcop

# Les variables escollides les guardem en aquest nou objecte:
dades <- stepwise$model
dades <- dades[c(2:ncol(dades),1)]


splitdf <- function(dataframe, seed = NULL){
  if(!is.null(seed)) set.seed(seed)
  index <- 1:nrow(dataframe)
  trainingindex <- sample(index,length(index)*0.8)
  trainingset <- dataframe[trainingindex, ]
  validateset <- dataframe[-trainingindex, ]
  list(trainingset = trainingset, validateset = validateset)
}

splits <- splitdf(dades, seed = 666)
# Ens dóna una llista: dos data frames anomenats "training" i "validate"
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
  rownames(Mesures) <- c("Accuracy","True Positive Rate",
                         "True Negative Rate", "Balanced Accuracy",
                         "Positive Predictive Value", "F1 score")
  colnames(Mesures) <- paste("Fold - ", 1:10)

  Mesures.mean <- apply(Mesures, 1, mean)
  Mesures.mean <- matrix(Mesures.mean, ncol = 1)

  rownames(Mesures.mean) <- rownames(Mesures)
  return(list(Confusion_matrix = matriu.confusio, Metrics = Mesures,
              Metrics_mean = Mesures.mean))
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

Naive <- Bayes_clasificator(training, wl, bl, 10, 666)
print(xtable(Naive$Metrics, caption="Mesures de comportament per cada fold del classificador Naive Bayes", label="tab:naive"), size="\\fontsize{7pt}{8pt}\\selectfont", table.placement="H")


####################################
#    Augmented Bayes classifier    #
####################################

atributes <- colnames(dades[-ncol(dades)])
wl <- data.frame(from = rep("target", length(atributes)), to = atributes)

Augmented <- Bayes_clasificator(training, wl, bl = NULL, 10, 666)
print(xtable(Augmented$Metrics, caption="Mesures de comportament per cada fold del classificador Augmented Naive Bayes", label="tab:augmented"),size="\\fontsize{7pt}{8pt}\\selectfont", table.placement="H")


###################################
#  Tria del clasificador bayesià  #
###################################

normalitat.naive <- shapiro.test(Naive$Metrics["Accuracy",])$p.value
normalitat.augmented <- shapiro.test(Augmented$Metrics["Accuracy",])$p.value

ttest <- t.test(Naive$Metrics["Accuracy",], Augmented$Metrics["Accuracy",],
                paired = TRUE, alternative = "two.sided")$p.value

Mesures <- cbind(Naive$Metrics_mean, Augmented$Metrics_mean)
colnames(Mesures) <- c("Naive Bayes", "Augmented Naive Bayes")

xtable(Mesures, table.placement="H")
print(xtable(Mesures, caption="Mitjanes de les mesures de comportament del classificador Naive Bayes i Augmented Naive Bayes", label="tab:mesures_globals"), table.placement="H")


#######################################################
# Augmented Bayes classifier training with validation #
#######################################################

atributes <- colnames(dades[-ncol(dades)])
wl <- data.frame(from = rep("target", length(atributes)), to = atributes)

xarxa <- hc(training, score = "bic", whitelist = wl)
xarxa.estimada <- bn.fit(xarxa, training, method = "mle")
xarxa.grain <- suppressWarnings(as.grain(xarxa.estimada))

prediccio <- NULL
CL <- NULL
prueba <- NULL
distribucio <- NULL

for(j in 1:nrow(test)){
  if(is.numeric(predict(xarxa.grain, response = "target",
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


matriu.confusio <- as.matrix(table(unlist(prediccio), test$target))
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
colnames(Mesures) <- "Mètriques"
print(xtable(Mesures, caption = "Mesures de comportament del classificador Augmented Naive Bayes amb l'aprenentatge sobre el conjunt d'entrenament i la validació sobre el de test", label = "tab:mesures_test"), table.placement="H")


#######################################################
# Augmented Bayes classifier training with validation #
#######################################################

atributes <- colnames(dades[-ncol(dades)])
wl <- data.frame(from = rep("target", length(atributes)), to = atributes)
xarxa.augmented <- hc(dades, score = "bic", whitelist = wl)


bl <- rbind(
  expand.grid(atributes, atributes),
  data.frame(Var1 = atributes, Var2= rep("target", length(atributes))),
  data.frame(Var1 = "target", Var2= "target")
)
xarxa.naive <- hc(dades, score = "bic", whitelist = wl, blacklist = bl)


# - Introduïm els nous casos de finals de partides d'escacs
naiv1 <- c(F,T,F,F,F,T,T,F,T,F,T,T,"g",F,"b",F,F,F,F,F,F,T,F,F,T,F,F,T,F,F,T,F,F,F,T,F,"n")
naiv1 <- naiv1[-c(5,13,19,24,26,28,29,36,37)]
naiv2 <- c(T,T,F,F,F,T,T,T,T,F,F,F,"n",T,F,F,F,F,F,F,T,T,F,F,F,F,F,T)

# - Introduim la white list i la black list, aprenem la xarxa Bayesiana Naive amb totes les dades:
xarxa.estimada.naive <- bn.fit(xarxa.naive, dades, method = "mle")
xarxa.grain.naive <- suppressWarnings(as.grain(xarxa.estimada.naive))

naiv_predictions <- function(var){
  xarxa.evid <- setEvidence(xarxa.grain.naive, nodes=atributes, states=var)
  evid <- xarxa.evid$evidence
  qq <- querygrain(xarxa.evid, nodes=c("target"), type="marginal")
  distribucio <- qq$target
  prediccio <- dimnames(distribucio)[[1]][which.max(distribucio)]
  CL <- round(100*max(distribucio), 2)
  reincidence.risc <- round(100*distribucio[2], 2)
  
  Resultat <- matrix(c(prediccio, CL, reincidence.risc), nrow=1)
  Resultat <- as.data.frame(Resultat)
  colnames(Resultat) <- c("Predicció",
                          "Confidence Level (CL) en %",
                          "Risc de reincidència")
  return(list("Evidències"=evid, "Resultat"=Resultat))
}

naiv_predictions(naiv1)
naiv_predictions(naiv2)


xtable(rbind("Jugada nova 1"=naiv_predictions(naiv1)$Resultat, 
             "Jugada nova 2"=naiv_predictions(naiv2)$Resultat),
       caption="Predicció de dos jugades noves amb el Naives Bayes entrenat",
       label="tab:chss3_pred",
       table.placement="H")


# AUGMENTED NAIVE BAYES
aug1 <- c(F,F,F,F,F,F,F,F,T,F,T,F,"l",F,"w",F,F,T,F,F,F,T,F,F,F,T,F,F,F,F,F,F,F,T,F,"n","won")
aug1 <- aug1[-c(5,13,19,24,26,28,29,36,37)]

aug2 <- c(F,F,F,F,F,F,T,F,F,F,T,F,"l",F,"b",F,F,T,F,F,F,F,F,F,F,T,F,F,F,F,F,F,F,T,T,"n","nowin")
aug2 <- aug2[-c(5,13,19,24,26,28,29,36,37)]

xarxa.estimada.augmented <- bn.fit(xarxa.augmented, dades, method = "mle")
xarxa.grain.augmented <- suppressWarnings(as.grain(xarxa.estimada.augmented))

aug_predictions <- function(var){
  xarxa.evid <- setEvidence(xarxa.grain.augmented, nodes=atributes, states=var)
  evid <- xarxa.evid$evidence
  qq <- querygrain(xarxa.evid, nodes=c("target"), type="marginal")
  distribucio <- qq$target
  prediccio <- dimnames(distribucio)[[1]][which.max(distribucio)]
  CL <- round(100*max(distribucio), 2)
  reincidence.risc <- round(100*distribucio[2], 2)
  
  Resultat <- matrix(c(prediccio, CL,reincidence.risc), nrow=1)
  Resultat <- as.data.frame(Resultat)
  colnames(Resultat) <- c("Predicció",
                          "Confidence Level (CL) en %",
                          "Risc de reincidència")
  return(list("Evidències"=evid, "Resultat"=Resultat))
}

aug_predictions(aug1)
aug_predictions(aug2)


xtable(rbind("Jugada nova 1"=aug_predictions(aug1)$Resultat, 
             "Jugada nova 2"=aug_predictions(aug2)$Resultat),
       caption="Predicció de dos jugades noves amb l'Augmented Naives Bayes entrenat",
       label="tab:chss4_pred",
       table.placement="H")


###########
## ANNEX ##
###########

#### Preprocessing de les dades: ####
print(xtable::xtable(model, caption="Model de regressió logística ajustat amb \texttt{glm}", label="tab:glm"), table.placement="H")


graphviz.plot(xarxa.augmented)


graphviz.plot(xarxa.naive)

