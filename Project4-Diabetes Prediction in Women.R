getwd()
setwd("C:/Users/HP/Documents/R Dataset")
rm(list = ls())
data <- read.csv("GL-Diabetes.csv", header = T)
view(data)

## Load Required Libraries
library(ggplot2)
library(car)
library(caret)
library(class)
library(devtools)
library(e1071)
library(Hmisc)
library(lmtest)
library(plyr)
library(pROC)
library(psych)
library(ROCR)
library(dplyr)
library(corrplot)
library(caTools)
library(DataExplorer)

names(data) <- c("NoPreg", "PlaGluConc", "DiastolicBP", "TSkinThick", "Test", "BMI", "DiabPediFunc", "Age", "Class")
dim(data)
str(data)
summary(data)
  describe(data)
attach(data)

prop.table(table(NoPreg,Class),1)*100


table(Class)
268/(500+268)

mystats <- function(x)
{
  nmiss<-sum(is.na(x))   #to calculate the missing values
  a <- x[!is.na(x)]      
  m <- mean(a)           #to calculate the mean
  n <- length(a)         #the length
  s <- sd(a)             #the standard devistion
  min <- min(a)          #the minimum value
  q1<-quantile(a,0.01)   #the different percentiles
  q5<-quantile(a,0.05)
  q95<-quantile(a,0.95)
  q99<-quantile(a,0.99)
  max <- max(a)          #the max value
  UC <- m+3*s            #the upper limit
  LC <- m-3*s            #the lower limit
  outlier_flag<- max>UC | min<LC #mark the variable/data with outlierflag, if it is above Upper cut-off/ lower than the Lower cut-off
  return(c(n=n, nmiss=nmiss, outlier_flag=outlier_flag, mean=m, stdev=s,min = min, q1=q1,q5=q5,q95=q95,q99=q99,max=max, UC=UC, LC=LC ))
}

#select the variables from the dataset, on which the calculations are to be performed.
diag_stats<-t(data.frame(apply(data[,c(1:9)], 2, mystats)))
View(diag_stats)

# Checking null data
sapply(data,function(x) sum(is.na(x)))
plot_missing(data)


# Checking for Outliers
par("mar")
par(mar=c(1,1,1,1))
boxplot(data[1:8], horizontal=TRUE,las=1, main="Diabetes Prediction")


# Checking # of unique values in each column
sapply(data,function(x) length(unique(x)))


# Target variable
length(which(data$Class=="1"))*100/nrow(data)

ggplot(data,aes(Class,fill = Class)) +
  geom_bar() + 
  ggtitle("Distribution of Outcome variable")


# Correlation check
dev.off()
library(corrplot)
cormat <- cor(data[,c(1:9)])
cor.plot(cormat)
diag (corMat) = 0 #Remove self correlations
plot_correlation(data[,c(1:9)])
corrplot.mixed(cormat,tl.pos = "lt")


#PLOT - Independent Variables vs Dependent Variable
##1. SCATTER PLOT
p1 <- ggplot(data, aes(x = PlaGluConc, y = DiabPediFunc)) +
  geom_point(aes(color=Class)) + 
  theme(legend.position = "bottom") +
  ggtitle("Relationship of Pregnancies with Age Vs Diabetes")

p2 <- ggplot(data,aes(x=Age, y=DiabPediFunc))+
  geom_point(aes(color=Class))+
  theme(legend.position = "bottom") +
  ggtitle("Relationship of Insulin with Glucose Vs Diabetes")

gridExtra::grid.arrange(p1, p2, ncol = 2)


## 2. SCATTER PLOT
p1 <- ggplot(data,aes(x=Age,y=Test))+
  geom_point(aes(color=Class))+
  theme(legend.position = "bottom") +
  ggtitle("Relationship of BMI with BP Vs Diabetes")

p2 <- ggplot(data,aes(x=BMI,y=TSkinThick))+
  geom_point(aes(color=Class))+
  theme(legend.position = "bottom") +
  ggtitle("Relationship of BMI with Skin Thickness Vs Diabetes")

gridExtra::grid.arrange(p1, p2, ncol = 2)

## 3. BOXPLOT
boxplot(NoPreg~Class)
boxplot(PlaGluConc~Class)
boxplot(DiastolicBP~Class)
boxplot(TSkinThick~Class)
boxplot(Test~Class)
boxplot(BMI~Class)
boxplot(DiabPediFunc~Class)
boxplot(Age~Class)


## Data Preparation
............................

## Split the Data
library(caret)
set.seed(1234)

pd <- sample(2,nrow(data), replace = T, prob = c(0.7,0.3))
train_data <- data[pd==1,]
test_data <- data[pd==2,]

prop.table(table(data$Class))
prop.table(table(train_data$Class))
prop.table(table(test_data$Class))

# Binary variables needs to be converted into factor variables
train_data$Class <- as.factor(train_data$Class)
test_data$Class <- as.factor(test_data$Class)

dim(train_data)
dim(test_data)

### Model Building - Logistic regression

logit_model1 = glm(Class ~ ., data = train_data, 
                   family = binomial)

summary(logit_model1)

# Check for multicollinearity
library(car)
vif(logit_model1)

#After removing Insignificant Variables
logit_model2 <- glm(Class ~ NoPreg+PlaGluConc+BMI+DiabPediFunc+DiastolicBP, data = train_data, 
                    family = binomial)

summary(logit_model2)

#Verify the best AIC and model
step_model <- step(logit_model1)


#Goodness of Fit
anova(logit_model1, test = "Chisq")


# Likelihood ratio test
library(lmtest)
lrtest(logit_model2)


# Pseudo R-square
library(pscl)
pR2(logit_model1)
1-(257.4688426/341.9401912)


# Odds Ratio
exp(coef(logit_model1))


# Probability
exp(coef(logit_model1))/(1+exp(coef(logit_model1)))


# Accuracy | Base Line Model
nrow(train_data[train_data$Class == 0,])/nrow(train_data)

# Performance metrics (train sample)
pred = predict(logit_model2, data=train_data, type="response")
y_pred_num = ifelse(pred>0.5,1,0)
y_pred = factor(y_pred_num, levels=c(0,1))
y_actual = train_data$Class
logi <- confusionMatrix(y_pred,train_data$Class,positive="1")

# Calibrating thresold levels to increase sensitivity
pred = predict(logit_model2, data=train_data, type="response")
y_pred_num = ifelse(pred>0.35,1,0)
y_pred = factor(y_pred_num, levels=c(0,1))
y_actual = train_data$Class
confusionMatrix(y_pred,y_actual,positive="1")

# Performance metrics (test sample)
pred = predict(logit_model2, newdata=test_data, type="response")
y_pred_num = ifelse(pred>0.35,1,0)
y_pred = factor(y_pred_num, levels=c(0,1))
y_actual = test_data$Class
confusionMatrix(y_pred,y_actual,positive="1")


# ROC plot
library(ROCR)
train_roc <- prediction(pred, train_data$Class)
dev.off()
plot(performance(train_roc, "tpr", "fpr"),
     col = "red", main = "ROC Curve for train data")
abline(0, 1, lty = 8, col = "blue")

# AUC
train_auc = performance(train_roc, "auc")
train_area = as.numeric(slot(train_auc, "y.values"))
train_area

# KS
ks_train <- performance(train_roc, "tpr", "fpr")
train_ks <- max(attr(ks_train, "y.values")[[1]] - (attr(ks_train, "x.values")[[1]]))
train_ks

# Gini
train_gini = (2 * train_area) - 1
train_gini



## KNN
# Normalize variables
scale = preProcess(train_data, method = "range")

train.norm.data = predict(scale, train_data)
test.norm.data = predict(scale, test_data)

knn_fit = train(Class ~., data = train.norm.data, method = "knn",
                trControl = trainControl(method = "cv", number = 3),
                tuneLength = 10)

knn_fit
knn_fit$bestTune$k
#knn_fit$results$Accuracy

#Plotting the k vs accuracy
plot((knn_fit$results$Accuracy)*100 - knn_fit$results$k, type = 'b', xlab = "# Neighbours", ylab = "Accuracy")

# Performance metrics (train sample)
pred = predict(knn_fit, data = train.norm.data[-9], type = "raw")
knnc <- confusionMatrix(pred,train.norm.data$Class,positive="1")

# Performance metrics (test sample)
pred = predict(knn_fit, newdata = test.norm.data[-9], type = "raw")
confusionMatrix(pred,test.norm.data$Class,positive="1")



### Model Building - NB

library(e1071)
NB = naiveBayes(x=train.norm.data[-c(9)], y=train.norm.data$Class)

# Performance metrics (train sample)
pred_NB_train = predict(NB, newdata = train.norm.data[-9])
nbc <- confusionMatrix(pred_NB_train, train.norm.data$Class,positive="1")

# Performance metrics (test sample)
pred_NB_test = predict(NB, newdata = test.norm.data[-9])
confusionMatrix(pred_NB_test,test.norm.data$Class,positive="1")


#Compare Accuracy
Model=c("Logistic Regression","KNN","Naive Bayes")
Accuracy<- c(logi$overall['Accuracy'], knnc$overall['Accuracy'], nbc$overall['Accuracy'])
accuracy <- data.frame(Model, Accuracy)
Accuracy<- c(logi, knnc, nbc)

ggplot(accuracy,aes(x=Model,y=Accuracy)) + geom_bar(stat='identity') + theme_bw()
      + ggtitle('Comparison of Model Accuracy')

names(data)
