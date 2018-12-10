set.seed(1)
setwd("C:\\Users\\Visagio\\Desktop\\Unicamp\\tcc")
library("FactoMineR")
library(tidyverse) # utility functions
library(caret) # hyperparameter tuning
library(randomForest) # for our model
library(Metrics) # handy evaluation functions
library(vegan)
library(caTools)
library(psych)
library(GPArotation)
library(MASS)
library(leaps)
library(ISLR)
library(FactoInvestigate)
library(ca)
library(factoextra)
library(leaps)
library(randomForest)
library(caret)


train=read.csv(file="train.csv", header = TRUE)#carregando base de dados
na.fail(train)#testando missing values, não deu erro então não tem NA
str(train)#Descrição das variáveis e seus tipos



train.f=train %>% select_if(is.factor)#selecione só os factors e crie train.f
train.f<-as.data.frame(model.matrix(~ ., data = train.f[,]))#abra os levels em variáveis
aa <- matrix(0L,nrow=dim(train.f)[2],ncol = 1)#crie uma matriz populada com zero de coluna 1 e linhas igual ao número de colunas em train.f
for(p in 1:dim(train.f)[2]){aa [p,1] <-sum(train.f[,p]!=0) }#substitua o zero da matiz aa pelo número de aparições em cada level
q <- 0.05 #percentagem mínima de dados para ser significante
#(Discutir esse valor)
aaa <- as.matrix(which(aa > q*4209)) # crie uma matriz com os números das colunas que atenda a significancia
train.f<-train.f %>% select_at(.var=aaa) #selecione apenas as variáveis com aparições mínimas
train.f<-lapply(train.f , factor)#transformar as colunas em fatores
train.f<-as.data.frame(train.f)[,-1]#Excluir Intercept

train.n <- train %>% select_if(is.numeric)
qqnorm(train$y)#Testando normalidade em y
qqline(train$y)#Leve problema nas caudas, mas considero passável
qqnorm(train$ID)#Testando normalidade do ID
qqline(train$ID)#Problema acentuado nas caudas
hist(train$ID, main= "Histograma ID") #Faz sentido termos uma produção que tende a ser constante e portanto seu histograma não representa uma distribuição normal
train.n2<-lapply(train.n[,3:dim(train.n)[2]] , factor)#transformar as colunas 3-370 em fatores
which(is.numeric(train.n2))#checar se todos são fatores
train.final <- cbind(train.n[,c(1,2)], train.f, train.n2)#Juntando tudo
train.final <- train.final[,-3]#tirando a intercept
str(train.final)
train.final.f <- train.final[,3:dim(train.final)[2]]


##Redução Dimensional
res.mca = MCA(train.final.f,ncp=48, graph = FALSE)
res.mca$eig#48=ncp representa 65%. Discutir isso!
eigen <- get_eigenvalue(res.mca)#separando os eigen
e.2<-as.data.frame(eigen)
nchar(row.names(e.2)[1])
e.1<-as.numeric(substr(row.names(e.2), 5, nchar(row.names(e.2))))
e.2<- as.data.frame(cbind(e.1,e.2))
qplot(x=e.2$e.1,y=e.2$cumulative.variance.percent,data=e.2, geom=c("point"), 
      main="Manutenção do banco", 
      xlab="Variáveis", ylab="Percentagem do banco mantida", abline(a = 48, b = 2, col = 2))


ind <- get_mca_ind(res.mca)#extraindo os valores
data <- as.data.frame(ind$coord)#extraindo valores
data<- cbind(train.final[,1:2],data)


data=as.data.frame(cbind(data$y,scale(data[,-2])))

##############################################################

##Separando treino e teste
sample = sample.split(data, SplitRatio = .66)
data.train = subset(data, sample == TRUE)
data.test  = subset(data, sample == FALSE)
attach(data.train)


#Modelar por combinatória até grupos de 31 variáveis
regfit.full=regsubsets(V1~.,data=data.train, nvmax = 31, really.big=FALSE)
names(regfit.full$force.in)


#Visualizar os resultados
reg.summary=summary(regfit.full)
par(mfrow=c(2,2)) # set R up to do the matrix plot (2 by 2)
reg.summary$rss
plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
points(which.min(reg.summary$rss),reg.summary$rss[which.min(reg.summary$rss)],col="red",cex=2,pch=20)
plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
points(which.max(reg.summary$adjr2),reg.summary$adjr2[which.max(reg.summary$adjr2)],col="red",cex=2,pch=20)
plot(reg.summary$cp, xlab = "Number of Variables", ylab = "CP", type = "l")
points(which.min(reg.summary$cp),reg.summary$cp[which.min(reg.summary$cp)],col="red",cex=2,pch=20)
plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type='l')
points(which.min(reg.summary$bic),reg.summary$bic[which.min(reg.summary$bic)],col="red",cex=2,pch=20)
regfit.fwd = regsubsets(V1 ~. , data=data.train,nvmax=50, method ="forward")
regfit.bwd = regsubsets(V1 ~. , data=data.train,nvmax=50,method ="backward")


plot(summary(regfit.fwd)$bic,xlab="Number of Variables",ylab="BIC",main= "Foward Selection",type='l')
points(which.min(summary(regfit.fwd)$bic),summary(regfit.fwd)$bic[which.min(summary(regfit.fwd)$bic)],col="red",cex=2,pch=20)

plot(summary(regfit.bwd)$bic,xlab="Number of Variables",ylab="BIC",main= "Backward Selection",type='l')
points(which.min(summary(regfit.bwd)$bic),summary(regfit.bwd)$bic[which.min(summary(regfit.bwd)$bic)],col="red",cex=2,pch=20)


test_mat = model.matrix (V1~., data = data.test)#preparando a matriz pro teste
val_errors = rep(NA,31)#armazenamento do teste

# Iterates over each size i
for(i in 1:31){
  
  # Extract the vector of predictors in the best fit model on i predictors
  coefi = coef(regfit.full, id = i)
  
  # Make predictions using matrix multiplication of the test matirx and the coefficients vector
  pred = test_mat[,names(coefi)]%*%coefi
  
  # Calculate the MSE
  val_errors[i] = mean((data.test$V1-pred)^2)
}


min = which.min(val_errors)# Find the model with the smallest error


plot(val_errors, type = 'b')# Plot the errors for each model size
points(min, val_errors[min][1], col = "red", cex = 2, pch = 20)


p=as.data.frame(coef(regfit.full,31))#Colocando os coeficientes do modelo de 31 variáveis em um data frame
w<-rownames(p)#extraindo o nome das linhas (nome das variáveis)
write.csv(w, file="Variáveis modelo.csv", row.names = FALSE)
modelo_regressivo=lm(as.formula(paste("V1 ~ ", paste(w[-1], collapse= "+"))), data=data.train)#fazendo o modelo regressivo, note que foi utilizado uma formula para colar os 31 nomes da variáveis no meio da fórmula
summary(modelo_regressivo)#obtendo os outputs do modelo


#Avaliando os Residuais
par(mfrow=c(1,1))
qqnorm(modelo_regressivo$residuals,datax = TRUE)
qqline(modelo_regressivo$residuals,datax = TRUE)
hist(modelo_regressivo$residuals)
plot(modelo_regressivo$residuals, main = "Erros Residuais")
reg.pred <- predict(modelo_regressivo, data.test[,-1])
reg.pred.train = predict(modelo_regressivo, data.train[,-1])
reg.Rmse.train = RMSE(pred=reg.pred.train, obs = data.train$V1)
reg.Rmse = RMSE(pred=reg.pred, obs = data.test$V1)

####### random Forest
tuned_model <- train(x = data.train[,-1], y = data.train[,1],
                     ntree = 5,
                     method = "rf") #Definir o mtry
print(tuned_model)
rf <- randomForest(x=data.train[,-1],data=data.train, y=data.train$V1, ntree=30, mtry=25)
par(mfrow=c(2,2))
plot(rf)
pred.rf<-predict(rf,newdata=data.test[,-1])
rf.rmse <- RMSE(pred=pred.rf, obs = data.test[,1])
pred.train = predict(rf, newdata = data.train[,-1])
rf.rmse.train <- RMSE(pred=pred.train ,obs = data.train[,1])
summary(rf)
val_errors.rf = mean((data.test$V1-pred.rf)^2)
val_errors[31]

