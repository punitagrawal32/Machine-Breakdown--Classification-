#### Problem Description 
# We have been given data from various machines. The features are 
# anonymized. We are required to predict if a machine will breakdown 
# or not. The features are mostly discrete, with 1 or 2 of them being 
# continuous. 


## 1. Setting the working directory and clearing the R environment

rm(list=ls(all=T))
setwd("C:/Users/Punit/Desktop")

## 2. Loading the required libraries 

library(RColorBrewer)
library(rattle)
library(ipred)
library(ROSE)
library(ada)
library(rpart.plot)
library(rpart)
library(randomForest)
library(C50)
library(factoextra)
library(xgboost)
library(glmnet)
library(mice)
library(dplyr)
library(ROCR)
library(DMwR)
library(car)
library(MASS)
library(vegan)
library(dummies)
library(infotheo)
library(caTools)
library(caret)
library(e1071)
library(corrplot)

## 3. Reading the data in R 

mach= read.csv("Machine_Data.csv", header= T, sep= ",")
mach2= read.csv("Machine_Data.csv", header= T, sep= ",",na.strings="-1")

View(mach)
View(mach2)

## 4. Data Exploration and Data Understanding


dim(mach2)
# 136399 x 24 
# 3273576 values 

str(mach2)
# Getting the structure of the data 

summary(mach2)
# Getting the summary of the data

sum(is.na(mach2))
# 4858 missing values

(sum(is.na(mach2))/3273576)*100
# 0.148% of the data is missing

table(mach2$Breakdown)
# 114705 0s and 21694 1s

(114705/136399)*100
# 84% of the data is 0 and 16% of the data is 1
# Great class imbalance

colSums(is.na(mach2))
# the missing values seem to be in V1, V3, V5, v14, v15, v18, and v21 

mac= read.csv("Machine_Data.csv")
mach_corr= cor(mac)
corrplot(mach_corr, method="color")

box= function(x){
  boxplot(x~ mac$Breakdown, main= "Boxplot ")
}

histo= function(y){
  barplot((table(y)), xlab=y, ylab="Frequency", main= "Barplot of variable distribution", col= "Black") 
}
histo(mac$V1)
attach(mac)
boxplot(Breakdown~, data=mac)

# Variable explanation 

# 1. id: this seems to be the transaction id. this is a variable with 
# unique numbers. Hence, we will remove this column in our analysis

table(mach2$id)
length(unique(mach2$id))

# 2. V1: Values ranging from 0 to 6. Might be a factor 

table(mach2$V1)
box(V1)
histo(V1)

# 3. V2: Values ranging from 0 to 7. More evenly distributed

table(mach2$V2)
box(V2)
histo(V2)

# 4. V3: Values ranging from 1 to 4

table(mach2$V3)
box(V3)
histo(V3)
#5. V4: Many levels. Going to let this be a numeric variable

table(mach2$V4)
box(V4)
histo(V4)

#6. V5: 0s and 1s only. Will convert this to a factor 

table(mach2$V5)
box(V5)
histo(V5)
#7. V6: 0s and 1s only. Will convert this to a factor 

table(mach2$V6)
box(V6)
histo(V6)

#8. V7: 0s and 1s only. Will convert this to a factor 

table(mach2$V7)
box(V7)
histo(V7)
#9. V8: 0s and 1s only. Will convert this to a factor 

table(mach2$V8)
box(V8)
histo(V8)
#10. V9: Many levels. Will let this stay numeric 

table(mach2$V9)
box(V9)
histo(V9)
#11. V10: 0s and 1s only. Will convert to a factor 

table(mach2$V10)
box(V10)
histo(V10)
#12. V11: 0s and 1s only. Will convert to a factor 

table(mach2$V11)
box(V11)
histo(V11)
#13. V12: Ranging from 0 to 0.9 

table(mach2$V12)
box(V12)
histo(V12)
#14. V13: Variable with continuous values 

table(mach2$V13)
box(V13)
histo(V13)
#15. V14: 0-11 values

table(mach2$V14)
box(V14)
histo(V14)
#16. V15: 0s and 1s only 

table(mach2$V15)
box(V15)
histo(V15)
#17. V16: values ranging from 0 to 9

table(mach2$V16)
box(V16)
histo(V16)
#18. V17: values ranging from 0 to 17

table(mach2$V17)
box(V17)
histo(V17)
#19. V18: Values ranging from 0 to 4

table(mach2$V18)
box(V18)
histo(V18)
#20. V19: Continuous variable

class(mach2$V19)
box(V19)
histo(V19)

#21. V20: Continuous variable 

table(mach2$V20)
box(V20)
histo(V20)
#22. V21: 0s and 1s only 

table(mach2$V21)
box(V21)
histo(V21)
#23. V22: 0 to 22 values

table(mach2$V22)
box(V22)
histo(V22)

#24. Breakdown: This is our target variable. 0 would mean that the
# machine did not break down and 1 would mean that the machine has 
# braoken down 

table(mach2$Breakdown)
box(Breakdown)
histo(Breakdown)

## 5. Data Cleaning 

mach= mach[,-1]
mach2=mach2[,-1]
# dropping the id column 

str(mach)
summary(mach)

str(mach2)
summary(mach2)

facto= function(x){
  x=as.factor(as.character(x))
}
names(mach)[1]

str(mach)

for(i in 1:ncol(mach)){
  if(names(mach)[i] %in% c("V5","V6","V7","V8","V10","V11","V15")){
    mach[,i]= facto(mach[,i])
  }
}


str(mach2)

for(i in 1:ncol(mach2)){
  if(names(mach2)[i] %in% c("V5","V6","V7","V8","V10","V11","V15","Breakdown")){
    mach2[,i]= facto(mach2[,i])
  }
}

# Instead of converting every variable manually, I defined a function
# to do the conversion. Thus, the required varibales have been
# converted.

which(duplicated(mach2))
# do duplicate values in the dataset

## Predicting missing values in mach2 using a decision tree 

colSums(is.na(mach2))
str(mach2)

V1fit<- rpart(V1 ~., data=mach2[!is.na(mach2$V1),], method="anova")
mach2$V1[is.na(mach2$V1)] <- predict(V1fit, mach2[is.na(mach2$V1),])

V3fit<- rpart(V3 ~., data=mach2[!is.na(mach2$V3),], method="anova")
mach2$V3[is.na(mach2$V3)] <- predict(V3fit, mach2[is.na(mach2$V3),])

V21fit<- rpart(V21 ~., data=mach2[!is.na(mach2$V21),], method="anova")
mach2$V21[is.na(mach2$V21)] <- predict(V21fit, mach2[is.na(mach2$V21),])

V18fit<- rpart(V18 ~., data=mach2[!is.na(mach2$V18),], method="anova")
mach2$V18[is.na(mach2$V18)] <- predict(V18fit, mach2[is.na(mach2$V18),])

V14fit<- rpart(V14 ~., data=mach2[!is.na(mach2$V14),], method="anova")
mach2$V14[is.na(mach2$V14)] <- predict(V14fit, mach2[is.na(mach2$V14),])

table(mach2$V5)
table(mach2$V15)

mach2[is.na(mach2$V15),]= 1
# mode imputation

mach2[is.na(mach2$V5),]= 1

colSums(is.na(mach2))

## 6. Train Test Split 

set.seed(125)
rows= createDataPartition(mach$Breakdown, p= 0.7, list= F)
train1= mach[rows, ] 
test1= mach[-rows,] 

train2= mach2[rows,]
test2= mach2[-rows,] 

## 7. Time for model building! 
# Model1: Decision Trees 

DT_rpart_Reg<-rpart(Breakdown~.,data=train1,method="class")
printcp(DT_rpart_Reg)
class(DT_rpart_Reg)
rpart.plot(DT_rpart_Reg)

str(train2)
str(test2)

predCartTrain=predict(DT_rpart_Reg, newdata=train1, type = "class")
predCartTest=predict(DT_rpart_Reg, newdata=test1, type= "class")
length(predCartTest)
length(predCartTrain)


confusionMatrix(predCartTrain, train1$Breakdown,positive = "1")
confusionMatrix(predCartTest, test1$Breakdown,positive = "1")

roc.curve(predCartTest, test1$Breakdown, plotit = F)

table(train2$Breakdown)

data.rose <- ROSE(Breakdown ~ ., data = train2, seed = 1)$data
table(data.rose$Breakdown)

DT_rpart_Reg<-rpart(Breakdown~.,data= data.rose,method="class", control = rpart.control(cp = 0.001))
printcp(DT_rpart_Reg)
fancyRpartPlot(DT_rpart_Reg)
predCartTrain=predict(DT_rpart_Reg, newdata=data.rose, type = "class")
predCartTest=predict(DT_rpart_Reg, newdata=test2, type= "class")
length(predCartTest)
length(predCartTrain)


confusionMatrix(predCartTrain, data.rose$Breakdown,positive = "1")
# accuracy62%, sensitvity68.11%, specificity55.67% 
confusionMatrix(predCartTest, test2$Breakdown,positive = "1")
# Accuracy74.45%, sensitivity 28.81%, specificty= 83.3%

roc.curve(predCartTest, test2$Breakdown, plotit = T)
# auc 54.9%


## Random Forest

model_rf = randomForest(Breakdown ~ . , data.rose, ntree = 200,mtry = 5)
summary(model_rf)
varImpPlot(model_rf)
rf_train_pred = predict(model_rf)
confusionMatrix(rf_train_pred, data.rose$Breakdown, positive = "1") 
# Accuracy65.2, sensitivity65.11%, specificity65.29
plot(model_rf)
preds_rf_test <- predict(model_rf, test2, type= "class")

confusionMatrix(preds_rf_test, test2$Breakdown, positive="1") 
# Accuracy81%, sensitivity 16.56%, specificity 92.18%

roc.curve(preds_rf_test, test2$Breakdown, plotit = T)
# 57.1% AUC

model_rf2 = randomForest(Breakdown ~ V19+V22+V9+V13+V4+V20+V17+V12+V2+V14 , data.rose, ntree = 200,mtry = 5)
summary(model_rf2)
varImpPlot(model_rf2)
rf_train_pred2 = predict(model_rf2)
confusionMatrix(rf_train_pred2, train2$Breakdown, positive = "1") 
# Accuracy65.2, sensitivity65.11%, specificity65.29

preds_rf_test2 <- predict(model_rf2, newdata=test2, type= "class")

confusionMatrix(preds_rf_test2, test2$Breakdown, positive="1") 
# Accuracy81%, sensitivity 16.56%, specificity 92.18%

roc.curve(preds_rf_test2, test2$Breakdown, plotit = T)
# 57.1% AUC




## KNN 

knn_model <- knn3(Breakdown ~ . , data.rose, k = 5)

knn_pred<- predict(knn_model, test2)
plot(knn_pred)
# * The predict function on the knn model returns probabilities for 
# each of the two classes in the target variable, so we'll get to the 
# class labels using the ifelse() function
knn_preds <- ifelse(knn_pred[, 1] > knn_pred[, 2], 0, 1)

confusionMatrix(knn_preds, test2$Breakdown, positive = "1")
# Accuracy67.26%, Sensitivity 32.56%, Specificity 73.82%
roc.curve(knn_preds, test2$Breakdown,plotit= T )
# 52.2% AUC 

# * Store the predictions on the train data

preds_train_k = predict(knn_model, data.rose ) 
preds_train_k
preds_train_knn = ifelse(preds_train_k[, 1] > preds_train_k[, 2], 0, 1)

confusionMatrix(preds_train_knn, data.rose$Breakdown, positive = "1")
# Accuracy72.26%, Sensitivity65%, Specificity80%
roc.curve(preds_train_knn, data.rose$Breakdown, plotit = T)
# AUC72.7%


## Bagging 

model_tree_bag <- bagging(Breakdown ~ . , data=data.rose,nbagg = 10,
                          control = rpart.control(cp = 0.01, xval = 10))
summary(model_tree_bag)
# * Test the model on the validation data and store the predictions on both the test and validation data

preds_tree_bag <- predict(model_tree_bag, test2)
confusionMatrix(preds_tree_bag, test2$Breakdown, positive = "1")
#Accuracy72.13%, Sensitivity28.71%, Specificity80.35%

roc.curve(preds_tree_bag, test2$Breakdown, plotit= T)
# 53.6%

preds_train_tree_bag <- predict(model_tree_bag)
confusionMatrix(preds_train_tree_bag, data.rose$Breakdown)
# Accuracy61.54%, Sensitivity66.20%, Specificity57%


roc.curve(preds_train_tree_bag, data.rose$Breakdown, plotit= T)
# 61.6%



## Stacking 

# Building a Stacked Ensemble

#  Before building a stacked ensemble model, we have to coallate all 
# the predictions on the train and validation datasets into a 
# dataframe

# Getting all the predictions on the train data into a dataframe

train_preds_df <- data.frame(rf = rf_train_pred, knn = preds_train_knn,
                             tree = predCartTrain,
                             Breakdown = data.rose$Breakdown )

# convert the target variable into a factor
train_preds_df$Breakdown <- as.factor(as.character(train_preds_df$Breakdown))


# * Use the sapply() function to convert all the variables other than 
# the target variable into a numeric type

numeric_st_df <- sapply(train_preds_df[, !(names(train_preds_df) %in% "Breakdown")], 
                        function(x) as.numeric(as.character(x)))

stackcor <-  cor(numeric_st_df)

corrplot(stackcor)
# * Now, since the outputs of the various models are not correlated 
# NO need to do pcalet's use PCA to reduce the dimensionality of the 
# dataset
# No correlation so not required 


# Now, add those columns to the target variable and convert it to a 
# data frame
stacked_df <- data.frame(numeric_st_df, Breakdown = train_preds_df$Breakdown)
str(stacked_df)
# * We will be building a logistic regression on the dataset to predict
# the final target variable

stacked_model <- glm(Breakdown ~ . , data = stacked_df,family = "binomial")

# Getting all the predictions from the validation data into a dataframe

#??? 
stack_df_test <- data.frame(rf = preds_rf_test, knn = knn_preds,
                            tree = predCartTest,
                            
                            Breakdown = test2$Breakdown)

dim(stack_df_test)
dim(stacked_df)
# Convert the target variable into a factor
stack_df_test$Breakdown <- as.factor(stack_df_test$Breakdown)
str(stack_df_test)
# Convert all other variables into numeric
numeric_st_df_test <- as.data.frame(  sapply(stack_df_test[, !(names(stack_df_test) %in% "Breakdown")],
                             function(x) as.numeric(as.character(x))))

str(numeric_st_df_test)
# * Now, apply the stacked model on the above dataframe
preds_st_test <-  predict(stacked_model, numeric_st_df_test,type = "response")
preds_st_test <- ifelse(preds_st_test > 0.5,"1","0")

# * Use the confusionMatrix() function from the caret package to get the evaluation metrics on the test data for the various models built today
confusionMatrix(preds_st_test, stack_df_test$Breakdown,positive = "1")
# Accuracy67.26%, Sensitivity32.56%, Specificity73.82%

roc.curve(preds_st_test, stack_df_test$Breakdown)
preds_st_train= predict( stacked_model, stacked_df,type= "response")
class(stacked_df)
preds_st_train= ifelse(preds_st_train>0.5,"1","0")
confusionMatrix(preds_st_train, stacked_df$Breakdown,positive = "1")
# Accuracy72.26%, Sensitivity65%, Specificity80%



## GBM
library(caret)
library(gbm)
str(train2)
class(train2)
train2= data.frame(train2)
test2= data.frame(test2)
trctrl<-trainControl(method = "cv", number = 3)
fit<-train(Breakdown~V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18,trControl=trctrl,method="gbm",data= train2,verbose = FALSE)
confusionMatrix(train2$Breakdown,predict(fit,train2),positive = "1")
gbmpre=predict(fit,train2)
# Accuracy84.13, Sensitivity80.29%, Specificity84%
gbmpred<-predict(fit,test2)
confusionMatrix(test2$Breakdown,gbmpred,positive = "1")
# Accuracy84.13%, Sensitivity75%, Specificity84.13%


## xGBoost
library(data.table)
library(mlr)

train2$Breakdown<-as.factor(as.character(train2$Breakdown))

setDT(train2)
setDT(test2)

library(mlr)
#using one hot encoding 

labels<-train2$Breakdown
ts_labels<-test2$Breakdown
str(train2)

numero= function(x){
  x=as.numeric(as.character(x))
}
names(mach)[1]

str(train2)

for(i in 1:ncol(train2)){
  if(names(train2)[i] %in% c("V5","V6","V7","V8","V10","V11","V15")){
    train2[,i]= numero(train2[,i])
  }
}

new_tr <- model.matrix(~.+0,data = train2[,-c("Breakdown"),with=F]) 
new_ts <- model.matrix(~.+0,data = test2[,-c("Breakdown"),with=F])  
labels <- as.numeric(labels)-1
ts_labels <- as.numeric(ts_labels)-1
library(xgboost)
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_labels)  
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, showsd = T, stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)
min(xgbcv$test.error.mean)
xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 79, watchlist = list(val=dtest,train=dtrain), print.every.n = 10, early.stop.round = 10, maximize = F , eval_metric = "error")
xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)
xgb2= predict(xgb1, dtrain)
xgbpred2 <- ifelse (xgb2 > 0.5,1,0)
confusionMatrix (xgbpred,ts_labels, positive="1")
# Accuracy84%, Sensitivity2%, Specificity99.4%
confusionMatrix (xgbpred2,labels ,positive="1")
#Accuracy84.94%, Sensitivity6%, Specificity99.9%
