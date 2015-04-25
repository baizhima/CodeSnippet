# edX-MIT Analytics Edge (Spring 2015): Kaggle Competition
# Author: Shan Lu (GitHub: baizhima) 
# Contact: shanlu at brown dot edu
# Affiliation: Brown University, Computer Science
# Date: Apr. 25th, 2015

# Set environment variables (locale to English)
Sys.setlocale("LC_ALL","C")
Sys.setenv(LANG="en")

### 1. Load libraries/data
# 1.1 Load libraries
library(tm)
library(SnowballC)
library(caTools)
library(rpart)
library(randomForest)

# 1.2 Load Training/Test datasets
# UniqueID: Train 1-6532, Test 6533-8402
setwd("~/Desktop/coursera/analyticsEdge/kaggle/")
Train = read.csv("NYTimesBlogTrain.csv",stringsAsFactors=FALSE)
Test = read.csv("NYTimesBlogTest.csv",stringsAsFactors=FALSE)
trainID = c(min(Train$UniqueID),max(Train$UniqueID))
testID = c(min(Test$UniqueID),max(Test$UniqueID))

# 1.3 Merge Train and Test set into Data
Test$Popular = 2
Data = rbind(Train, Test)


### 2. Build Abstract Corpus
# 2.1 Build Headline Corpus
corpusH = Corpus(VectorSource(Data$Headline))
corpusH = tm_map(corpusH, tolower)
corpusH = tm_map(corpusH, PlainTextDocument)
corpusH = tm_map(corpusH, removePunctuation)
corpusH = tm_map(corpusH, removeWords, stopwords("english"))
corpusH = tm_map(corpusH, stemDocument)
frequenciesH = DocumentTermMatrix(corpusH)
sparseH = removeSparseTerms(frequenciesH, 0.995) # 142 words remained
HeadlineSparse = as.data.frame(as.matrix(sparseH))
colnames(HeadlineSparse) = paste0("h_",colnames(HeadlineSparse))
HeadlineSparse$UniqueID = Data$UniqueID

# 2.2 Build Abstract Corpus
corpusA = Corpus(VectorSource(Data$Abstract))
corpusA = tm_map(corpusA, tolower)
corpusA = tm_map(corpusA, PlainTextDocument)
corpusA = tm_map(corpusA, removePunctuation)
corpusA = tm_map(corpusA, removeWords, stopwords("english"))
corpusA = tm_map(corpusA, stemDocument)
frequenciesA = DocumentTermMatrix(corpusA)
sparseA = removeSparseTerms(frequenciesA, 0.99) # 223 words remained
AbstractSparse = as.data.frame(as.matrix(sparseA))
colnames(AbstractSparse) = paste0("a_",colnames(AbstractSparse))
AbstractSparse$UniqueID = Data$UniqueID

### 3. Build cleaned data for modeling

# 3.1 Merge words from Headline and Abstract corpus together
wordSparse = merge(HeadlineSparse, AbstractSparse, by="UniqueID")
allData = merge(Data, wordSparse, by="UniqueID")

# 3.2 Ignore useless variables
allData$Abstract = NULL
allData$Snippet = NULL
allData$Headline = NULL 

# 3.3 Factorize section/subsetction name strings
allData$NewsDesk = as.factor(allData$NewsDesk)
allData$SectionName = as.factor(allData$SectionName)
allData$SubsectionName = as.factor(allData$SubsectionName)
#allData$Popular = as.factor(allData$Popular)

# 3.4 Extract Year, Month, Day, HH/MM/SS from $PubDate
allData$PubDate = strptime(allData$PubDate,format="%Y-%m-%d %H:%M:%S")
allData$Weekday = as.factor(weekdays(allData$PubDate))
allData$Hour = as.factor(allData$PubDate$hour)
allData$PubDate = NULL

# 3.5 split word counts into different levels by quartiles
quartiles = summary(allData$WordCount)
wordsLevel = c(quartiles["1st Qu."],quartiles["Median"],quartiles["3rd Qu."]) 
allData$LenType = as.integer(allData$WordCount > wordsLevel[1])
allData$LenType = allData$LenType + as.integer(allData$WordCount > wordsLevel[2])
allData$LenType = allData$LenType + as.integer(allData$WordCount > wordsLevel[3])
allData$LenType = as.factor(allData$LenType)
allData$WordCount = NULL

# 3.6 Rearrange column orders by name
allData = allData[, order(names(allData))] 

# 3.7 Split Train/Test data by Popular
blogTrain = subset(allData, Popular <= 1)
blogTest = subset(allData, Popular == 2)
blogTest$Popular = NULL # to be predicted


### 4. Model 1: Classification Tree(CART)
# 4.1 Build CART model
modelCART = rpart(Popular ~ . - UniqueID, data=blogTrain,method='class')
# 4.2 Make predictions on training data
predictCART = predict(modelCART, data=blogTrain, type='class')
confusionCART = table(blogTrain$Popular, predictCART) # training set accuracy 0.906
trainAccuracyCART = (confusionCART[1,1]+confusionCART[2,2])/sum(confusionCART)
# 4.3 Write predictions into CSV file "predictCART.csv"
resultCART = as.data.frame(blogTest$UniqueID)
colnames(resultCART)[1] = "UniqueID"
resultCART$Probability1 = predict(modelCART, newdata=blogTest)[,2]
write.table(resultCART, "predictCART.csv", sep=",",row.names=FALSE)

### 5. Model 2: Random Forest


