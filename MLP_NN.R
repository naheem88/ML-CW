library(readxl)
library(neuralnet)
library(Metrics)

# Read the data set
exchangeUSD_df = read_xlsx("ExchangeUSD.xlsx")

# Extract exchange rates data
exchangeRates = exchangeUSD_df$`USD/EUR`

# Normalize function (Min-max normalization)
normalize = function(data) {
  normalized_data = (data - min(data)) / (max(data) - min(data))
  return(normalized_data)
}

# Un-normalize function
unnormalize <- function(x, min, max) {
  return( (max - min)*x + min )
}

# Split data into training and testing sets
trainData = exchangeRates[1:400]
testData = exchangeRates[401:500]

# Input vector (t-1)
t1Train = cbind(D_previous = lag(trainData, 1), D_current = trainData)
t1Test = cbind(D_previous = lag(testData, 1), D_current = testData)

# Input vector (t-2)
t2Train = cbind(D_previous2 = lag(trainData, 2), D_previous = lag(trainData, 1),
                    D_current = trainData)
t2Test = cbind(D_previous2 = lag(testData, 2), D_previous = lag(testData, 1),
                   D_current = testData)

t3Train = cbind(D_previous3 = lag(trainData, 3),
                    D_previous2 = lag(trainData, 2),
                    D_previous = lag(trainData, 1),
                    D_current = trainData)
t3Test = cbind(D_previous3 = lag(testData, 3),
                   D_previous2 = lag(testData, 2),
                   D_previous = lag(testData, 1),
                   D_current = testData)

t4Train = cbind(D_previous4 = lag(trainData, 4),
                    D_previous3 = lag(trainData, 3),
                    D_previous2 = lag(trainData, 2),
                    D_previous = lag(trainData, 1),
                    D_current = trainData)
t4Test = cbind(D_previous4 = lag(testData, 4),
                   D_previous3 = lag(testData, 3),
                   D_previous2 = lag(testData, 2),
                   D_previous = lag(testData, 1),
                   D_current = testData)

# Removes rows with missing data and normalize
t1Train = t1Train[complete.cases(t1Train), ]
t1TrainNorm = normalize(as.data.frame(t1Train))
t1Test = t1Test[complete.cases(t1Test), ]
t1TestNorm = normalize(as.data.frame(t1Test))

t2Train = t2Train[complete.cases(t2Train), ]
t2TrainNorm = normalize(as.data.frame(t2Train))
t2Test = t2Test[complete.cases(t2Test), ]
t2TestNorm = normalize(as.data.frame(t2Test))

t3Train = t3Train[complete.cases(t3Train), ]
t3TrainNorm = normalize(as.data.frame(t3Train))
t3Test = t3Test[complete.cases(t3Test), ]
t3TestNorm = normalize(as.data.frame(t3Test))

t4Train = t4Train[complete.cases(t4Train), ]
t4TrainNorm = normalize(as.data.frame(t4Train))
t4Test = t4Test[complete.cases(t4Test), ]
t4TestNorm = normalize(as.data.frame(t4Test))

# Column names for I/O
colnames(t1TrainNorm) = c("previousDay", "currentDay")
colnames(t1TestNorm) = c("previousDay", "currentDay")

colnames(t2TrainNorm) = c("DayBefore2", "previousDay", "currentDay")
colnames(t2TestNorm) = c("DayBefore2", "previousDay", "currentDay")

colnames(t3TrainNorm) = c("DayBefore3", "DayBefore2", "previousDay", "currentDay")
colnames(t3TestNorm) = c("DayBefore3", "DayBefore2", "previousDay", "currentDay")

colnames(t4TrainNorm) = c("DayBefore4", "DayBefore3", "DayBefore2", "previousDay", "currentDay")
colnames(t4TestNorm) = c("DayBefore4", "DayBefore3", "DayBefore2", "previousDay", "currentDay")

evaluate_model <- function(y_expected, y_pred) {
  rmse = rmse(unlist(y_expected), y_pred)
  mae = mae(y_expected, y_pred)
  mape = mape(y_expected, y_pred)
  smape = smape(y_expected, y_pred)
  return(list(rmse = rmse, mae = mae, mape = mape, smape = smape))
}

model_train <- function(tNum, hiddenNum, linearBoolean) {
  set.seed(98)
  if (tNum == 1) {
    model = neuralnet(currentDay ~ previousDay, data = t1TrainNorm, hidden = hiddenNum,
                       linear.output = linearBoolean)
    
    predicted_Target = predict(model, t1TestNorm)
    predicted_Target = unnormalize(predicted_Target, min(t1Test), max(t1Test))
    y_expected = unnormalize(t1TestNorm$currentDay, min(t1Test), max(t1Test))
  } else if (tNum == 2) {
    model = neuralnet(currentDay ~ previousDay + DayBefore2, data = t2TrainNorm, 
                      hidden = hiddenNum, linear.output = linearBoolean)
    
    predicted_Target = predict(model, t2TestNorm)
    predicted_Target = unnormalize(predicted_Target, min(t2Test), max(t2Test))
    y_expected = unnormalize(t2TestNorm$currentDay, min(t2Test), max(t2Test))
  } else if (tNum == 3) {
    model = neuralnet(currentDay ~ previousDay + DayBefore2 + DayBefore3, 
                      data = t3TrainNorm, hidden = hiddenNum, linear.output = linearBoolean)
    
    predicted_Target = predict(model, t3TestNorm)
    predicted_Target = unnormalize(predicted_Target, min(t3Test), max(t3Test))
    y_expected = unnormalize(t3TestNorm$currentDay, min(t3Test), max(t3Test))
  } else if (tNum == 4) {
    model = neuralnet(currentDay ~ previousDay + DayBefore2 + DayBefore3 + DayBefore4, 
                       data = t4TrainNorm, hidden = hiddenNum, linear.output = linearBoolean)
    
    predicted_Target = predict(model, t4TestNorm)
    predicted_Target = unnormalize(predicted_Target, min(t4Test), max(t4Test))
    y_expected = unnormalize(t4TestNorm$currentDay, min(t4Test), max(t4Test))
  } else {
    stop("Invalid input vector number")
  }
  metrics = evaluate_model(y_expected, predicted_Target)
  print(metrics)
}

print("Model 1")
model_train(1, 4, TRUE)

print("Model 2")
model_train(2, 4, TRUE)

print("Model 3")
model_train(3, 4, TRUE)

print("Model 4")
model_train(4, 4, TRUE)

print("Model 5")
model_train(1, 4, FALSE)

print("Model 6")
model_train(2, 4, FALSE)

print("Model 7")
model_train(3, 4, FALSE)

print("Model 8")
model_train(4, 4, FALSE)

print("Model 9")
model_train(1, c(1,1), TRUE)

print("Model 10")
model_train(2, c(1,4), TRUE)

print("Model 11")
model_train(3, c(1,9), TRUE)

print("Model 12")
model_train(4, c(2,8), TRUE)

print("Model 13")
model_train(2, 14, TRUE)

print("Model 14")
model_train(3, 14, TRUE)

print("Model 15")
model_train(4, 14, TRUE)

set.seed(98)
best_model = neuralnet(currentDay ~ previousDay + DayBefore2 + DayBefore3, 
                  data = t3TrainNorm, hidden = 14, 
                  linear.output = TRUE)

predicted_Target = predict(best_model, t3TestNorm)
predicted_Target = unnormalize(predicted_Target, min(t3Test), max(t3Test))
expected = unnormalize(t3TestNorm$currentDay, min(t3Test), max(t3Test))
metrics = evaluate_model(expected, predicted_Target)
print(metrics)
plot(best_model)

par(mfrow=c(1,1))
plot(expected, predicted_Target, col='red', xlab = "Actual Rate", ylab = "Predicted Rate", main='Real vs predicted NN', pch=18, cex=0.7) 
abline(a=0, b=1, h=90, v=90) 
