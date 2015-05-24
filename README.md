# Practical Machine Learning
### by SMinkes

## Summary

Velloso, E. et all (http://groupware.les.inf.puc-rio.br/)  state that the quality of executing an activity that recevied only little attention thus far. They have the opinion that the quality of executing an event provides potentially usefull information. They investigate three aspects that pertain to qualitative activity recognition: specifying correct execution, detecting execution mistakes, providing feedback on the to the user. Velloso, E. et all try out a sensor- and model-based approach to qualitatitive recognition of activities. 

Our aim in this paper is to classify the activites in five different classes (A, B, C, D and E). We investigate three models using the data of Velloso, E. et all: A Decistion Tree model, A Random Forest model and a Naive Bayes model. The Random Forest model outperforms (Accuracy = 99%) both the Decision Tree model (Accuracy = 51%) and the Naive Bayes model (Accuracy = 73%).

## Setup

Loading all the required packages.

```{r echo=TRUE, results='hide', message=FALSE, warning=FALSE}
library(caret)
library(rattle)
library(rpart)
library(e1071)
library(rpart.plot)
library(randomForest)
library(gridExtra)
library(knitr)
```

## Getting and Cleaning Data

Let's read in the csv

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE}
data <- read.csv("pml-training.csv", header=TRUE, sep=",",  na.strings=c("NA","#DIV/0!",""))
outofsample <- read.csv("pml-testing.csv", header=TRUE, sep=",",  na.strings=c("NA","#DIV/0!",""))
```

## Explorative Data Analysis

The most interesting variable is our classification variable "Classe", which is our dependent variable.
```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE}
summary(data$classe)
```

The classification variable seems to be more or less evenly spread around the five classes. No further transformations are necessary. For now we want to select a good set of predictors (which are our independent variables). Lets first check the proportion of missings ("NAs")

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE}
nac <- numeric() 
for (i in 1:ncol(data)){ nac[i] <- sum(is.na(data[,i]))}
NAdf <- data.frame(table(round(nac/dim(data)[1], 3) * 100))
names(NAdf) <- c("Percent NAs", "Number of Columns")
NAdf
```

Many columns have more than 90% missing values. We cannot use these columns as viable predicters, thus we remove them. Also the first 8 columns, such as a time_stamp, do not have a meaningful prediction value. We remove these columns as well. Let's select for our out of sample set exactly the same columns. Since the main focus is on modeling in this Course Project, we leave the Explorative Data Analysis for now.

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE}
trainClasse <- data[, nac == 0]
trainClasse <- trainClasse[, 9:60]
trainClasse$Classe <- data[ ,160]
```

## Splitting the dataset in a Training data set and a Test (Cross validation) data set

We opt for a 60/40 split.

```{r, results='hide', message=FALSE, warning=FALSE}
set.seed(123)
inTrain = createDataPartition(trainClasse$Classe, p = .60, list=FALSE)
training= trainClasse[ inTrain,]
test = trainClasse[-inTrain,]
```

## Modeling

We need to classifiy a categorical variable with 5 different classes. A Decision Tree model is well suited for this purpose. Lets train our Decision Tree model!


```{r, results='hide', message=FALSE, warning=FALSE}
DecTreFit <- train(Classe~.,method="rpart", data=training, trControl=trainControl(method = "cv", number = 4))
```

To have some insight in our Decision tree, let's create an Decision Tree plot with fancyRpartPlot.

```{r, results='hide', message=FALSE, warning=FALSE}
## Create fancy plot
fancyRpartPlot(DecTreFit$finalModel,cex=.5,under.cex=1,shadow.offset=0)
```

And let's actually predict our model on the Test (Cross Validation) set!

```{r, results='hide', message=FALSE, warning=FALSE}
DecTrePred=predict(DecTreFit,test)
confusionMatrix(test$Classe,DecTrePred)
```

The accuracy seems to be only 53%, which seems to be a bit disappointing. A Decision Tree model is definitely not a good option for this data. Probably a different type of model is more accurate. Lets see if a Random Forest model improves our accuracy! Let's train it!

```{r, results='hide', message=FALSE, warning=FALSE}
## Random Forest
RanForFit <- train(Classe ~ ., method="rf", trControl=trainControl(method = "cv", number = 4), data=training)
```

And lets predict!

```{r, results='hide', message=FALSE, warning=FALSE}
RanForPred=predict(RanForFit,test)
confusionMatrix(test$Classe,RanForPred)
```

The accuracy is 99%! This is suspiciously high. As a result our in sample error is just 1%. Scoring such high accuracy makes the out of sample error (which is higher than the in sample error by definition) very interesting. Let's first try a third model and see how that works. We opt for a Naive Bayes model, let's train it again!

```{r, results='hide', message=FALSE, warning=FALSE}
NaiBayFit <- train(Classe ~ ., data=training, method="nb", trControl=trainControl(method = "cv", number = 4))
```

And of course, lets predict our test set!

```{r, results='hide', message=FALSE, warning=FALSE}
NaiBayPred=predict(NaiBayFit,test)
confusionMatrix(test$Classe, NaiBayPred)
```

The Naive Bayes set scored an accuracy of 73%, which is reasonable. 

## Model Evaluation

Let's evaluate our models and compare them in one table. The Random Forest model is by far the best (99% Accuracy), followed by a Naive Bayes model (73% Accuracy) and a third place for the Decision Tree model (51% Accuracy).

```{r, echo=TRUE, message=FALSE, warning=FALSE}
## Let's compare
compare <- list("Decision Tree"=DecTreFit$results[1, 2:5], 
                "Random Forest"=RanForFit$results[1, 2:5], 
                "Naive Bayes"=NaiBayFit$results[2, 3:6])
compare <- do.call(rbind.data.frame, compare)
compare$OutOfSampleError <- 1- c( as.numeric(confusionMatrix(test$Classe, DecTrePred)[[3]][1]),
                                  as.numeric(confusionMatrix(test$Classe, RanForPred)[[3]][1]),
                                  as.numeric(confusionMatrix(test$Classe, NaiBayPred)[[3]][1]))
kable(compare)
```

Finally, let's predict our out of sample set. 

```{r, echo=TRUE, message=FALSE, warning=FALSE}
result <- predict(RanForFit, outofsample)

result
```

Creating the files

```{r}
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(result)
```

