####################
## Prediction des modes de transport
####################
tab=read.table("Mode_app4.txt",header=T)
View(tab)

#on crée une nouvelle variable binaire (train/autre)
tab$choix2=as.factor(tab$choix=="train")
levels(tab$choix2)=c("autre","train")
View(tab)




#modèle logistique pour choix2
fit0=glm(choix2~.-choix,data=tab,family="binomial") #avec toutes les variables
summary(fit0)
#sélection Backward : on enlève la variable ayant la p-value la plus élevée
fit=glm(choix2~.-choix-tps_voit,data=tab,family="binomial") 
summary(fit)
#on continue ainsi pas à pas pour obtenir
fit2=glm(choix2~.-choix-tps_voit-prix_bus-prix_covoit-prix_train,data=tab,family="binomial")
summary(fit2) #tout est significatif
anova(fit2,fit0,test="Chisq") #toutes les contraintes sont acceptées par rapport au modèle global

#on peut aussi appliquer une sélection automatique basée sur BIC
n=nrow(tab)
tmp=step(fit,k=log(n))
tmp$call #la formule du modèle sélectionnée est identique à celui retenu par backward
fit_train=glm(tmp$call,data=tab,family="binomial") #idem que fit2




#Probabilités estimées par cross-validation (LOO)
pred=1:n #initialisation
for(i in 1:n){
  fit=glm(tmp$call,data=tab,family="binomial",subset=-i) #on estime le modèle sans i
  pred[i]=predict(fit,tab[i,-1],type="response") #on prédit la proba associée à i
}



#On en déduit la courbe ROC
library(ROCR)
pr = prediction(pred, tab$choix2) #prévisions (par LOO) contre vraies valeurs : nécessaires pour les calculs de "performance" suivants
roc = performance(pr, measure = "tpr", x.measure = "fpr") #calcul des points de la courbe ROC (en ayant utilisé le LOO)
plot(roc)




#Calcul du nombre de mal classés en fonction du seuil
res.err=performance(pr, measure = "err")
plot(res.err) #on observe que le minimum se trouve vers 0.45

#Pour récupérer le minimum, il faut aller dans l'objet res.err qui est de type S4 sous R.
#Les éléments d'un objet S4 sont des "slots" auxquels on accède avec @ (l'équivalent de $ pour les listes)
slotNames(res.err)
seuil=unlist(res.err@x.values)[which.min(unlist(res.err@y.values))]
seuil

table(pred>seuil, Ex18$train)

#Validation
tab_valid=read.table("Mode_valid4.txt",header=T)
View(tab_valid)


#On prédit les probabilités en utilisant le modèle précédent
pred=predict(fit_train,tab_valid,type="response")

pred

#matrice de confusion 
#on commence par créer choix2
tab_valid$choix2=as.factor(tab_valid$choix=="train")
levels(tab_valid$choix2)=c("autre","train")
#on en déduit la table de contingence (matrice de confusion)
table(tab_valid$choix2,pred>seuil)



##Modélisation des 4 classes
library(nnet)
#modèle avec toutes les variables (ne pas oublier d'enlever choix2)
fit=multinom(choix~.-choix2,data=tab)
coef(fit)


#sélection automatique par BIC
tmp=step(fit,k=log(n))
fit_sel=multinom(tmp$call,data=tab)
summary(fit_sel)

#Prévision des modalités sur Mode_valid
pred4=predict(fit_sel,tab_valid) 
#ici l'option type="response" n'est pas mise car on prédit directement les modalités selon la proba la plus élevée (on ne choisit pas le seuil)
# Complément : de plus, si on voulait la prévision des probas associées à chaque modalité, il faudrait mettre l'option type="probs" et non type="response"
# (car l'objet fit_sel est de class multinom et non glm, et donc la fonction predict ne se décline pas de la même manière)
predict(fit_sel,tab_valid,type="probs") #pour exemple

#Matrice de confusion
table(tab_valid[,1],pred4)
