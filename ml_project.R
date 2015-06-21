#' Initially we preprocess the data by removing columns with more than 90% NAs
#' so we have better predictors for our model. To give credit where it is due, 
#' the idea of removing columns with lots of NA is discussed in the forums about 
#' this assignment, I adopted it as it sounds like a good suggestion. We also 
#' remove columns unlike to be meaningful, again tip from the same article. 

# Removes columns with high NA
removeHighNA <- function(frame)
{
  goodCols = c()
  cols <- names(frame)
  for (col in cols)
  {
    if (percentNA(frame[,col]) < 0.90) 
    {
      goodCols <- cbind(goodCols,col)
    }
  }
  
  frame[,goodCols]
}

# Displays percent of NAs in each column of a frame 
percentNAFrame <- function(frame)
{
  cols <- names(frame)
  for (col in cols)
  {
    print(paste(col, percentNA(frame[,col])))
  }
}

# Returns percent of NAs in a column
percentNA <- function(x) 
{
  sum(is.na(x)==TRUE)/length(x)
}

# Load data, specifying both  NA and DIV/0 as NAs
pmlTraining <- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!"))

# Clean data, removing columns with too many NAs and uninteresting columns
cleanedData <- removeHighNA(pmlTraining)
cleanedData <- subset(cleanedData,select=-raw_timestamp_part_1)
cleanedData <- subset(cleanedData,select=-raw_timestamp_part_2)
cleanedData <- subset(cleanedData,select=-cvtd_timestamp)
cleanedData <- subset(cleanedData,select=-user_name)
cleanedData <- subset(cleanedData,select=-X)

# Save the name of the columns being used, so we can use it to subset the test set
# to predict
goodColumns <- names(cleanedData)

#' We also enable multi-core processing to make computation time reasonable.
#' Another great suggestion from the forums.
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

#' Now we partition the training set into a training set proper and a test or
#' *quiz* set. We have enough samples, and setting aside a test set will allow 
#' us to have a better estimate of accuracy. 

require("caret")
# Separate a *quiz* subset from the training data, so we can use it to 
# evaluate out of set error
set.seed(13445)
inTrain <- createDataPartition(y=cleanedData$classe, p=0.75, list=FALSE)
training <- cleanedData[inTrain,]
quiz <- cleanedData[-inTrain,]

#' We use 5-fold cross validation to estimate the accuracy. We chose to 
#' use random forests as it is a very accurate classifier. 
fitControl <- trainControl(
    method = "cv",
    number = 5, 
    allowParallel = TRUE)

set.seed(138)
cvFit <- train(classe ~ ., 
    data = training,
    trControl = fitControl,
    method = "rf")

#' Here we can see the error estimate from the cross-validation
#' classifier. With the chosen mtry the Accuracy is 99.76%.
cvFit

#' Perform final fit using the whole set.
set.seed(138)
finalFit <- train(classe ~ ., 
               data = training,
               method = "rf")

# Stop cluster processing to shutdown processes and free memory
stopCluster(cl)

#' With the whole set, mtry chosen is still 28, and the in-sample
#' accuracy is 99.60%.
finalFit

#' Now we use the quiz set to perform a validation and get a measure
#' of out-of-sample error, to make sure we are not over-fitting. Accuracy
#' in this set is 99.82%, so seems like this was a good fit.
predictedAnswers <- predict(finalFit, newdata = quiz)
print("Error estimated on *quiz* set")
sum(predictedAnswers == quiz$classe)/length(predictedAnswers)

#' Now we load the test set and generate final answers using the test set.
# Load data, specifying both  NA and DIV/0 as NAs
pmlTesting <- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!"))

# Subset testing data to select same columns
preProcessed <- pmlTesting[,names(pmlTesting) %in% goodColumns]

# Predict outcomes 
finalAnswers <- predict(finalFit, newdata = preProcessed)

# Write submission files
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(finalAnswers)
