### Datamining en Finance et Assurance - Flora ZIADI

# on importe les librairies 
library(knitr)
library(DMwR)
library(RGtk2)
library(glmnet)
library(e1071)
library(neuralnet)
library(class) # librairie pour le KNN
library(rpart) #librairie pour les arbres de d�cision
library(randomForest)  #librairie pour les for�ts al�atoires

# library(rattle)
# rattle()

set.seed(123)


# on importe les donn�es
data = read.table("imports-85.txt", 
                  header = FALSE, 
                  sep = ",", 
                  col.names=c("symboling", "normalized-losses", "make",
                              "fuel_type", "aspiration", "num_of_doors",
                              "body-style", "drive-wheels", "engine-location",
                              "wheel-base", "length", "width",
                              "height", "curb-weight", "engine-type",
                              "num-of-cylinders", "engine-size", "fuel-system",
                              "bore", "stroke", "compression-ratio", 
                              "horsepower", "peak-rpm","city-mpg",
                              "highway-mpg", "price"))  

summary(data)

write.csv(data, file="C:/Users/flora_000/Desktop/ENSAE/Semestre 2/Datamining en Assurance/data.csv",row.names=TRUE) 

### Affectation du bon type aux donn�es



data_clean = as.data.frame(data)

data_clean$bore              = as.numeric(as.character(data_clean$bore))
data_clean$stroke            = as.numeric(as.character(data_clean$stroke))
data_clean$horsepower        = as.numeric(as.character(data_clean$horsepower))
data_clean$peak.rpm          = as.numeric(as.character(data_clean$peak.rpm))
data_clean$price             = as.numeric(as.character(data_clean$price))

data_clean$num_of_doors[data_clean$num_of_doors == "?"] = NA

data_clean = data_clean[ , (names(data_clean) != "normalized.losses")]




### Remplacement des valeurs manquantes


data_complete = knnImputation(data_clean, k = 10, meth = "median")

print(paste("Lignes initiales avec des valeurs manquantes :", 
nrow(data_clean[!complete.cases(data_clean),])))

print(paste("Lignes avec des valeurs manquantes apr�s remplissage :", 
nrow(data_complete[!complete.cases(data_complete),])))

summary(data_complete)
### Transformation des variables cat�gorielles

dummy = function(dataframe, column){
# Ajout des colonnes dummies
# on retire la premi�re colonne pour ne pas avoir de colonnes colin�aires redondantes
df = cbind(dataframe, 
model.matrix(~dataframe[ , (names(dataframe) == column)] - 1)[, -1]) 

# Suppression de la colonne d'origine devenue superflue
df = df[ , (names(df) != column)]

# Nommage clean des colonnes
names(df) = gsub("dataframe.*\\]", 
                 paste0(column, "."), 
                 names(df))
names(df) = gsub("model.*", 
                 paste0(column, "."), 
                 names(df))
names(df) = gsub("-", 
                 paste0(""), 
                 names(df))

return(df)
}

# Parcours des colonnes cat�gorielles pour remplacement
data_num = data_complete
column_types = sapply(data_num, class)
for (column in names(data_num)) {
  if (column_types[column] == "factor") {
    data_num = dummy(data_num, column)
  }
}

print(paste("Nouveau nombre de features (toutes num�riques) =", ncol(data_num) - 1))
print("Variables :")
matrix(colnames(data_num), ncol=3)


### Normalisation des donn�es

data_norm = data_num
for (i in 2:ncol(data_norm)) {
  data_norm[, i] = (data_norm[, i] - mean(data_norm[, i])) / sd(data_norm[, i])
}



### S�paration apprentissage/test

test_num = 55

# Extraction des features et labels dans des matrices/vecteurs,
# pour certains algorithmes qui ne supportent pas les dataframes
x <- model.matrix(symboling ~ ., data_norm)[,-1]
y <- data_norm$symboling

indice_train = sample(1:nrow(x), nrow(x) - test_num)
indice_test = (-indice_train)
x_train = x[indice_train,]
y_train = y[indice_train]
x_test  = x[indice_test,]
y_test  = y[indice_test]


# Partie 2 : Mod�lisation


## R�gression lin�aire + r�gularisation L2 (ridge)


# Param�tres lambda � tester
lambda_cv = 10^seq(10, -2, length = 100)

# OLS simple
OLS_simple = lm(symboling ~ ., data = data_num[indice_train,])
summary(OLS_simple)
y_pred_OLS_simple = predict(OLS_simple, data_num[indice_test,-1])
MSE_OLS_simple = round(mean((y_pred_OLS_simple - y_test)^2), 3)

print(paste("MSE de la r�gression sans r�gularisation :", MSE_OLS_simple))

anova(OLS_simple)
# on affiche la matrice de confusion
y_pred_reg = round(y_pred_OLS_simple)
table(y_pred_reg,y_test)

# on trace la courbe des valeurs pr�dites contre les valeurs observ�es :
plot(y_pred_OLS_simple,  y_test, main = "R�gression simple", xlab = "Predictions", ylab = "True Values")
abline(0, 1, lty = 2)

# OLS r�gularis�e (k-fold avec k=10 pour identifier les param�tres)
OLS_reg = glmnet(x_train, y_train, alpha = 0, lambda = lambda_cv)
OLS_cv = cv.glmnet(x_train, y_train, alpha = 0)

best_lambda = OLS_cv$lambda.min
y_pred_OLS_reg = predict(OLS_reg, s = best_lambda, newx = x_test)

plot(OLS_cv)

MSE_OLS_reg = round(mean((y_pred_OLS_reg - y_test)^2), 3)
print(paste("MSE de la r�gression r�gularis�e :", MSE_OLS_reg))

# on trace la courbe des valeurs pr�dites contre les valeurs observ�es :
plot(y_pred_OLS_reg,  y_test, main = "R�gularisation L2 (ridge)", xlab = "Predictions", ylab = "True Values")
abline(0, 1, lty = 2)

# on affiche la matrice de confusion
table(round(y_pred_OLS_reg),y_test)


## KNN


## KNN avec le param�tre k par d�faut
y_pred = as.numeric(as.character(knn(x_train,x_test, y_train, k = 3)))
# on affiche la matrice de confusion
table(y_pred,y_test)
# on calcule le MSE
MSE_KNN_simple = round(mean((y_pred - y_test)^2), 3)
print(paste("MSE du KNN avec des param�tres par d�faut :", MSE_KNN_simple))

set.seed(123)
## KNN avec des param�tres optimis�s
# Pour trouver le k optimal, nous utilisons une validation crois�e (5-fold cross-validation)
# Nous testons k entre 1 et 10
fold = sample(rep(1:5,each=41)) # creation des groupes B_v
cvpred = matrix(NA,nrow=205,ncol=10) # initialisation de la matrice
MSE_V_fold = matrix(NA,nrow=5,ncol=10) # initialisation de la matrice
MSE_k_tot = rep(0,10)
MSE_k_moy = rep(0,10)
# des pr�dicteurs
for (k in 1:10)
{ 
for (v in 1:5)
{
  v_x_train = x[which(fold != v),]
  v_x_test = x[which(fold == v),]
  v_y_train = y[which(fold != v)]
  cvpred[which(fold == v),k] = as.numeric(as.character(knn(v_x_train,v_x_test,v_y_train,k = k)))
  # on calcule le MSE pour chaque v
  v_y_test = y[which(fold == v)]
  MSE_V_fold[v,k] = round(mean((cvpred[which(fold == v),k] - v_y_test)^2), 3)
  # on calcule le MSE pour chaque k
  MSE_k_tot[k] = MSE_k_tot[k] + MSE_V_fold[v,k]
}
} 

k = seq(1, 10, by = 1)
MSE_k_moy[k] = MSE_k_tot[k] / 5
plot(k, MSE_k_moy[k], type = "l", col = "red", xlab = "k", ylab = "MSE")

# Mod�le final
y_pred_knn = as.numeric(as.character(knn(x_train,x_test, y_train, k = 1)))
# on affiche la matrice de confusion
table(y_pred_knn,y_test)
# on calcule le MSE
MSE_KNN_tune = round(mean((y_pred_knn - y_test)^2), 3)
print(paste("MSE du KNN avec des param�tres optimis�s:", MSE_KNN_tune))

# on trace la courbe des valeurs pr�dites contre les valeurs observ�es :
plot(y_pred_knn,  y_test, main = "KNN", xlab = "Predictions", ylab = "True Values")
abline(0, 1, lty = 2)


## SVM

svm_simple = svm(symboling ~ ., data = data_norm[indice_train,])
y_pred = predict(svm_simple, data_norm[indice_test,-1])
MSE_SVM_simple = round(mean((y_pred - y_test)^2), 3)
print(paste("MSE du SVM avec des param�tres par d�faut :", MSE_SVM_simple))

svm_opti <- tune(svm, 
                 symboling ~ .,
                 data = data_norm[indice_train,],
                 ranges = list(cost    = 16^(0:3), # r�gularisation
                               kernel  = c("polynomial", "radial", "sigmoid"))
)
print(svm_opti)

y_pred_SVM = predict(svm_opti$best.model, data_norm[indice_test,-1])
MSE_SVM_tune = round(mean((y_pred_SVM  - y_test)^2), 3)
print(paste("MSE du SVM optimis� :", MSE_SVM_tune))

# on affiche la matrice de confusion
table(round(y_pred_SVM) ,y_test)
# on trace la courbe des valeurs pr�dites contre les valeurs observ�es :
plot(y_pred_SVM ,  y_test, main = "SVM (param�tres optimis�s)", xlab = "Predictions", ylab = "True Values")
abline(0, 1, lty = 2)


## Arbre de d�cision


## Arbre de d�cision avec les param�tres par d�faut

# Dans la fonction rpart, 'data' doit �tre au format data.frame, 
# elle n'accepte ni les tableaux, ni les matrices.
df_x_train = data.frame(x_train)
df_x_test = data.frame(x_test)

# Etape 1 : d�finition de l'arbre et apprentissage 
rt = rpart(y_train ~ ., data = df_x_train) 

# on affiche l'arbre de d�cision   
plot(rt, compress = TRUE,main="Regression Tree (param�tres par d�faut)")
text(rt, use.n = TRUE, col = "blue")

# Etape 2 : pr�diction
y_pred = predict(rt, df_x_test) # pr�diction de l'arbe de d�cision

# Etape 3 : evaluation 
MSE_CART_simple = regr.eval(y_test, y_pred,"mse")
# MSE_CART_simple = round(mean((y_pred - y_test)^2), 3) les 2 expressions sont identiques
print(paste("MSE du CART avec des param�tres par d�faut :", MSE_CART_simple))
# on trace la courbe des valeurs pr�dites contre les valeurs observ�es :
plot(y_pred,  y_test, main = "Regression Tree (param�tres par d�faut)", 
     xlab = "Predictions", ylab = "True Values")
abline(0, 1, lty = 2)

## Arbre de d�cision avec des param�tres optimis�s
# on souhaite faire varier les param�tres suivants :
# cp : ce param�tre intervient en pr�-�lagage lors de la construction de l'arbre, 
#      une segmentation est accept�e uniquement si la r�duction relative de l'indice 
#      de Gini est sup�rieure � � cp �.
# minsplit(par d�faut = 20) : le nombre minimal d'observations qui doivent exister 
#      dans un noud pour qu'une tentative de division soit effectu�e.
# maxdepth(par d�faut = 30) : La profondeur maximale de l'arbre.
# minbucket : le nombre minimum d'observations dans tout noeud <leaf> terminal. 
#      Si un seul de minbucket ou minsplit est sp�cifi�, le code d�finit minsplit 
#      sur minbucket * 3 ou minbucket sur minsplit / 3, selon le cas.
# Etape 1 : recherche du mod�le optimal
rt_opti <- tune( rpart, 
                 symboling ~ .,
                 data = data_norm[indice_train,],
                 ranges = list( minsplit =seq(10,30, by = 1), 
                                maxdepth =seq(10,30, by = 1),
                                cp =seq(0.01,0.20, by = 0.02)) )

print(rt_opti)
print("Le mod�le optimal est obtenu avec les param�tres : minsplit=15, maxdepth= 10, cp=0.01")
# on affiche l'arbre de d�cision 
rt = rpart(y_train ~ ., data = df_x_train,minsplit=15, maxdepth= 10, cp=0.01) 
plot(rt, compress = TRUE,main="Regression Tree (param�tres optimis�s)")
text(rt, col = "blue")

# Etape 2 : pr�diction
y_pred_cart = predict(rt_opti$best.model, data_norm[indice_test,-1])

# Etape 3 : evaluation 
MSE_CART_tune = round(mean((y_pred_cart - y_test)^2), 3)
print(paste("MSE du CART optimis� :", MSE_CART_tune))
# on trace la courbe des valeurs pr�dites contre les valeurs observ�es :
plot(y_pred_cart,  y_test, main = "Regression Tree (param�tres optimis�s)", 
     xlab = "Predictions", ylab = "True Values")
abline(0, 1, lty = 2)

# on affiche la matrice de confusion
table(round(y_pred_cart),y_test)

## For�t al�atoire

set.seed(123)
df_x_train = data.frame(x_train)
df_x_test = data.frame(x_test)

## For�t al�atoire avec les param�tres par d�faut

# Etape 1 : d�finition de l'arbre et apprentissage 
rf = randomForest(y_train ~ ., data = df_x_train)

# Etape 2 : pr�diction
y_pred = predict(rf, df_x_test)

# Etape 3 : evaluation 
MSE_RF_simple = round(mean((y_pred - y_test)^2), 3) 
print(paste("MSE de la for�t al�atoire avec des param�tres par d�faut :", MSE_RF_simple))
# on trace la courbe des valeurs pr�dites contre les valeurs observ�es :
plot(y_pred,  y_test, main = "Random Forest (param�tres par d�faut)",
     xlab = "Predictions", ylab = "True Values")
abline(0, 1, lty = 2)

## For�t al�atoire avec des param�tres optimis�s
# on souhaite faire varier les param�tres suivants :
# ntree : Le nombre d'arbres dans la for�t. (par d�faut = 500)
# mtry : le nombre de variables test�es � chaque division 
#        (par d�faut = racine carr� du nombre de colonne, donc ici 8)
# Etape 1 : recherche du mod�le optimal
set.seed(123)
rf_opti <- tune( randomForest, 
                 symboling ~ .,
                 data = data_norm[indice_train,],
                 ranges = list( ntree = seq(200,800, by = 50), 
                                mtry = seq(5,20, by = 1)) )

print(rf_opti)
# best parameters: ntree = 500,  mtry = 13
# Etape 2 : pr�diction
y_pred_RF = predict(rf_opti$best.model, data_norm[indice_test,-1])

# Etape 3 : evaluation
MSE_RF_tune = round(mean((y_pred_RF - y_test)^2), 3)
print(paste("MSE de la for�t al�atoire optimis� :", MSE_RF_tune))
# on trace la courbe des valeurs pr�dites contre les valeurs observ�es :
plot(y_pred_RF,  y_test, main = "Random Forest (param�tres optimis�s)", 
     xlab = "Predictions", ylab = "True Values")
abline(0, 1, lty = 2)


## R�seau de neurones


set.seed(123)
n = names(data_norm)
f = as.formula(paste("symboling ~ ", 
paste(n[!n %in% "symboling"], 
collapse = " + ")))

nn_10 = neuralnet(f,
data=data_norm[indice_train,],
hidden = c(10),
linear.output = T)
nn_20 = neuralnet(f,
data=data_norm[indice_train,],
hidden = c(20),
linear.output = T)
nn_40 = neuralnet(f,
data=data_norm[indice_train,],
hidden = c(40),
linear.output = T)
nn_10_10 = neuralnet(f,
data=data_norm[indice_train,],
hidden = c(10, 10),
linear.output = T)
nn_20_10 = neuralnet(f,
data=data_norm[indice_train,],
hidden = c(20, 10),
linear.output = T)
nn_10_10_10 = neuralnet(f,
data=data_norm[indice_train,],
hidden = c(10, 10, 10),
linear.output = T)


y_pred_NN_10 = round(compute(nn_10, data_norm[indice_test,-1])$net.result)
MSE_NN_10 = round(mean((y_pred_NN_10 - y_test)^2), 3)
print(paste("MSE du NN avec 1 couche cach�e de 10 neurones :", MSE_NN_10))

y_pred_NN_20 = round(compute(nn_20, data_norm[indice_test,-1])$net.result)
MSE_NN_20 = round(mean((y_pred_NN_20 - y_test)^2), 3)
print(paste("MSE du NN avec 1 couche cach�e de 20 neurones :", MSE_NN_20))

y_pred_NN_40 = round(compute(nn_40, data_norm[indice_test,-1])$net.result)
MSE_NN_40 = round(mean((y_pred_NN_40 - y_test)^2), 3)
print(paste("MSE du NN avec 1 couche cach�e de 40 neurones :", MSE_NN_40))

y_pred_NN_10_10 = round(compute(nn_10_10, data_norm[indice_test,-1])$net.result)
MSE_NN_10_10 = round(mean((y_pred_NN_10_10 - y_test)^2), 3)
print(paste("MSE du NN avec 2 couches cach�es de 10 neurones :", MSE_NN_10_10))

y_pred_NN_20_10 = round(compute(nn_20_10, data_norm[indice_test,-1])$net.result)
MSE_NN_20_10 = round(mean((y_pred_NN_20_10 - y_test)^2), 3)
print(paste("MSE du NN avec 2 couches cach�es de 20 et 10 neurones :", MSE_NN_20_10))

y_pred_NN_10_10_10 = compute(nn_10_10_10, data_norm[indice_test,-1])$net.result
MSE_NN_10_10_10 = round(mean((y_pred_NN_10_10_10  - y_test)^2), 3)
print(paste("MSE du NN avec 3 couches cach�es de 10 neurones :", MSE_NN_10_10_10))

# on trace la courbe des valeurs pr�dites contre les valeurs observ�es :
plot(y_pred_NN_10_10_10,  y_test, main = "RN avec 3 couches cach�es de 10 neurones ", 
     xlab = "Predictions", ylab = "True Values")
abline(0, 1, lty = 2)

# on affiche la matrice de confusion
table(round(y_pred_NN_10_10_10),y_test)


## valeurs observ�es
## R�gression lin�aire 
print(paste('R�gression lin�aire : ', MSE_OLS_simple))
## R�gularisation L2 (ridge)
print(paste('R�gularisation L2 (ridge) : ',MSE_OLS_reg))
## KNN
print(paste('KNN : ',MSE_KNN_tune))
## Arbre de d�cision
print(paste('Arbre de d�cision : ',MSE_CART_tune))
## For�t al�atoire
print(paste('For�t al�atoire : ',MSE_RF_tune ))
## SVM 
print(paste('SVM : ',MSE_SVM_tune))
## R�seau de neurones
print(paste('R�seau de neurones: ',MSE_NN_10_10_10))


