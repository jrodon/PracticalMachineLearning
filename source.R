library(dplyr)
library(caret)
library(corrplot)
library(parallel)
library(doParallel)
## Load the data
test.Data <- read.csv("~/data_science/R_Programming/Course8Week4Project/pml-testing.csv",
                      na.strings = c("","NA","#DIV/0!"))
train.Data <- read.csv("~/data_science/R_Programming/Course8Week4Project/pml-training.csv",
                       na.strings = c("","NA","#DIV/0!"))
## Data exploration
dim(test.Data)
dim(train.Data)
names(train.Data[which(names(test.Data)!=names(train.Data))]) # Differing names
names(test.Data[which(names(test.Data)!=names(train.Data))])
summary(train.Data) # Shows many NA
summary(test.Data) # Shows many NA
sum(apply(test.Data,2,function(x) mean(is.na(x)) > 0.5)) # Useless variables

## Cleaning the data
### Eliminate columns full of NAs, or with more than 50% of them
NA.Cols <- apply(test.Data,2,function(x) mean(is.na(x))) > 0.5
train.Data <- train.Data[,!NA.Cols]
test.Data <- test.Data[,!NA.Cols]
### Eliminate columns with near-zero variance
NZV.Cols <- nearZeroVar(train.Data, foreach = T)
train.Data <- train.Data[,-NZV.Cols]
test.Data <- test.Data[,-NZV.Cols]
### Eliminate predictors that are independent of the outcome
train.Data <- train.Data[,-c(1:6)]
test.Data <- test.Data[,-c(1:6)]

# ## Outliers
# train.Data$accel_forearm_y[which(train.Data$accel_forearm_y > 900)] <- NA
# train.Data$gyros_forearm_x[which(train.Data$gyros_forearm_x < -20)] <- NA
# train.Data$gyros_forearm_y[which(train.Data$gyros_forearm_y > 300)] <- NA
# train.Data$gyros_forearm_z[which(train.Data$gyros_forearm_z > 200)] <- NA
# train.Data$gyros_dumbbell_x[which(train.Data$gyros_dumbbell_x < -200)] <- NA
# train.Data$gyros_dumbbell_y[which(train.Data$gyros_dumbbell_y > 4)] <- NA
# train.Data$gyros_dumbbell_z[which(train.Data$gyros_dumbbell_z > 300)] <- NA
# train.Data$accel_dumbbell_x[which(train.Data$accel_dumbbell_x < -400)] <- NA
# train.Data$magnet_dumbbell_y[which(train.Data$magnet_dumbbell_y < -3000)] <- NA

### Split the data
set.seed(1780)
in.Train <- createDataPartition(train.Data$classe, p = 0.75, list = F)
split.train.Data <- train.Data[in.Train,]
split.test.Data <- train.Data[-in.Train,]

## Explore the variables
### Correlation matrix
cor.M <- cor(split.train.Data[,-53]) 
corrplot(abs(cor.M), title = "Correlation Plot", method = "square", 
         order = "hclust", type = "lower", diag = F,
         tl.col = "black", tl.cex = 0.8, tl.srt = 30,
         cl.lim = c(0,1), cl.cex = 0.8, cl.ratio = 0.1)

## Create the model
### Parallelize
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

CV.control <- trainControl(method = "cv", number = 10, allowParallel = T, verboseIter = T)
model.Fit.RF <- train(classe ~ ., data = split.train.Data, 
                      method = "rf", trControl = CV.control)
model.Fit.CART <- train(classe ~ ., data = split.train.Data, 
                        method = "rpart", trControl = CV.control)

stopCluster(cluster)
registerDoSEQ()

### PCA
# corr.M <- abs(cor(CV.Train.Data[,-53]))
# diag(corr.M) <- 0
# high.Corr <- which(corr.M > 0.8, arr.ind = T)
# high.Corr <- high.Corr[!duplicated(t(apply(high.Corr,1,sort))),]
# 
# pre.Proc <- preProcess(CV.Train.Data[,unique(high.Corr[,1])], method = c("pca"), pcaComp = 2)
# PCA.CV.Train.Data <- predict(pre.Proc,CV.Train.Data[,unique(high.Corr[,1])])
# qplot(PC1,PC2,data = PCA.CV.Train.Data, color = CV.Train.Data$classe)

### Model results
model.Fit.RF$finalModel
confusionMatrix(split.train.Data$classe,predict(model.Fit.RF,split.train.Data))

### Change cols type
# test.Data$cvtd_timestamp <- as.POSIXct(test.Data$cvtd_timestamp, format = "%d/%m/%Y %R")
# train.Data$cvtd_timestamp <- as.POSIXct(train.Data$cvtd_timestamp, format = "%d/%m/%Y %R")
# 
# char.Cols <- lapply(train.Data,class)
# head(train.Data[,char.Cols])
# train.Data[,char.Cols]
# names(train.Data[,lapply(train.Data,class)=="factor"])
## Exploring the data
library(ggplot2)

qplot(roll_forearm, roll_arm, data = CV.Train.Data, color = classe)
qplot(pitch_forearm, pitch_arm, data = train.Data, color = classe)
qplot(yaw_forearm, yaw_arm, data = train.Data, color = classe)
qplot(total_accel_forearm, total_accel_arm, data = train.Data, color = classe)

qplot(roll_dumbbell, roll_forearm, data = train.Data, color = classe)
qplot(pitch_dumbbell, roll_forearm, data = train.Data, color = classe)
qplot(yaw_dumbbell, yaw_forearm, data = train.Data, color = classe)
qplot(total_accel_dumbbell, total_accel_forearm, data = train.Data, color = classe)

featurePlot(data_frame(CV.Train.Data$total_accel_belt,
                       CV.Train.Data$total_accel_arm,
                       CV.Train.Data$total_accel_dumbbell,
                       CV.Train.Data$total_accel_forearm),
            CV.Train.Data$classe, plot = "box")

featurePlot(data_frame(CV.Train.Data$pitch_belt,
                       CV.Train.Data$pitch_arm,
                       CV.Train.Data$pitch_dumbbell,
                       CV.Train.Data$pitch_forearm),
            CV.Train.Data$classe, plot = "box")

featurePlot(data_frame(CV.Train.Data$roll_belt,
                       CV.Train.Data$roll_arm,
                       CV.Train.Data$roll_dumbbell,
                       CV.Train.Data$roll_forearm),
            CV.Train.Data$classe, plot = "box")

featurePlot(data_frame(CV.Train.Data$yaw_belt,
                       CV.Train.Data$yaw_arm,
                       CV.Train.Data$yaw_dumbbell,
                       CV.Train.Data$yaw_forearm),
            CV.Train.Data$classe, plot = "box")

qplot(accel_belt_x, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = accel_belt_y, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = accel_belt_z, data = train.Data, color = classe) # Could be used
qplot(y = gyros_belt_x, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = gyros_belt_y, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = gyros_belt_z, data = train.Data, color = classe) # Could be used
qplot(y = magnet_belt_x, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = magnet_belt_y, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = magnet_belt_z, data = train.Data, color = classe) # Could be used

qplot(y = accel_arm_x, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = accel_arm_y, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = accel_arm_z, data = train.Data, color = classe) # Could be used
qplot(y = gyros_arm_x, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = gyros_arm_y, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = gyros_arm_z, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = magnet_arm_x, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = magnet_arm_y, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = magnet_arm_z, data = train.Data, color = classe, facets = user_name ~ .)

qplot(y = accel_forearm_x, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = accel_forearm_y, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = accel_forearm_z, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = gyros_forearm_x, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = gyros_forearm_y, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = gyros_forearm_z, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = magnet_forearm_x, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = magnet_forearm_y, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = magnet_forearm_z, data = train.Data, color = classe, facets = user_name ~ .)

qplot(y = accel_dumbbell_x, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = accel_dumbbell_y, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = accel_dumbbell_z, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = gyros_dumbbell_x, data = train.Data, color = classe, facets = user_name ~ .)
qplot(y = gyros_dumbbell_y, data = train.Data, color = classe) # Could be used
qplot(y = gyros_dumbbell_z, data = train.Data, color = classe)
qplot(y = magnet_dumbbell_x, data = train.Data, color = classe)
qplot(y = magnet_dumbbell_y, data = train.Data, color = classe)
qplot(y = magnet_dumbbell_z, data = train.Data, color = classe, facets = user_name ~ .)

qplot(gyros_dumbbell_x, gyros_dumbbell_z, data = train.Data, color = classe, facets = user_name ~ classe) + 
      geom_smooth(method = "lm")
qplot(total_accel_forearm^2, (accel_forearm_x/10)^2 + (accel_forearm_y/10)^2 + (accel_forearm_z/10)^2, data = train.Data, color = classe)
qplot(total_accel_dumbbell, accel_dumbbell_z, data = train.Data, color = classe, alpha = 0.7)
qplot(accel_arm_z^2, accel_arm_y^2/accel_arm_z^2, data = train.Data, color = classe, log = "xy")
qplot(y=accel_forearm_x^2,data = train.Data, color = classe)
qplot(y=roll_dumbbell/pitch_dumbbell,data = train.Data, color = classe)
