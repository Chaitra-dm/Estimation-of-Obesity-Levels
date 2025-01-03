# Load libraries
library(factoextra)
library(cluster)
library(outliers)
library(class)
library(caret)
library(proxy)

# Read the dataset
obesity_data <- read.csv("//Users//ravi//Desktop//Mining//ObesityDataSet_raw_and_data_sinthetic.csv", stringsAsFactors = TRUE)

print(obesity_data$NObeyesdad)

# Check the structure and summary of the dataset
head(obesity_data, 3)
str(obesity_data)
summary(obesity_data)

# Checking missing values in dataset
colSums(is.na(obesity_data))

#Convert categorical variable into factor
obesity_data$Gender = factor(obesity_data$Gender, levels = c('Female','Male'), labels = c(1,2))
obesity_data$family_history_with_overweight = factor(obesity_data$family_history_with_overweight, levels = c('yes','no'), labels = c(1,0))

obesity_data$FAVC = factor(obesity_data$FAVC, levels = c('yes','no'), labels = c(1,0))
obesity_data$SMOKE = factor(obesity_data$SMOKE, levels = c('yes','no'), labels = c(1,0))
obesity_data$SCC = factor(obesity_data$SCC, levels = c('yes','no'), labels = c(1,0))

obesity_data$CAEC = factor(obesity_data$CAEC, levels = c('no', 'Sometimes','Frequently','Always'), labels = c(0,1,2,3))
obesity_data$CALC = factor(obesity_data$CALC, levels = c('no', 'Sometimes','Frequently','Always'), labels = c(0,1,2,3))

obesity_data$MTRANS = factor(obesity_data$MTRANS, levels = c('Public_Trans', 'Walking','Automobile','Motorbike','Bike'), labels = c(1,2,3,4,5))

obesity_data$NObeyesdad = factor(obesity_data$NObeyesdad, levels = c('Insufficient_Weight', 'Normal_Weight','Overweight_Level_I','Overweight_Level_II', 'Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'), labels = c(1,2,3,4,5,6,7))

#Rounding up 
obesity_data$Age = as.numeric(format(round(obesity_data$Age, 0)))
obesity_data$FCVC = as.numeric(format(round(obesity_data$FCVC, 0)))
obesity_data$NCP = as.numeric(format(round(obesity_data$NCP, 0)))
obesity_data$CH2O = as.numeric(format(round(obesity_data$CH2O, 0)))
obesity_data$FAF = as.numeric(format(round(obesity_data$FAF, 0)))
obesity_data$TUE = as.numeric(format(round(obesity_data$TUE, 0)))

##convert to numeric variable
obesity_data$MTRANS <- as.numeric(obesity_data$MTRANS)
obesity_data$CALC <- as.numeric(obesity_data$CALC)

obesity_data$SCC <- as.numeric(obesity_data$SCC)
obesity_data$SMOKE <- as.numeric(obesity_data$SMOKE)

obesity_data$CAEC <- as.numeric(obesity_data$CAEC)
obesity_data$FAVC <- as.numeric(obesity_data$FAVC)

obesity_data$family_history_with_overweight <- as.numeric(obesity_data$family_history_with_overweight)
obesity_data$Gender <- as.numeric(obesity_data$Gender)

obesity_data$NObeyesdad <- as.numeric(obesity_data$NObeyesdad)

str(obesity_data)

#Outlier Analysis
age_outlier <- grubbs.test(obesity_data$Age)
age_outlier     

#Outlier Analysis
Height_outlier <- grubbs.test(obesity_data$Height)
Height_outlier 

#Outlier Analysis
Weight_outlier <- grubbs.test(obesity_data$Weight)
Weight_outlier 

############ KNN Algoritham

#split data
#Setting seed

set.seed(5000)
ind <- sample(2, nrow(obesity_data), replace = T, prob = c(0.7, 0.3))
train <- obesity_data[ind == 1,]
test <- obesity_data[ind == 2,]

# Extract target variable
y <- obesity_data$NObeyesdad

# Define the number of neighbors for KNN
k <- 5  # You can choose any value for k

# Train KNN model
knn_model <- knn(train = train, test = test, cl = y[ind == 1], k = k)

# Confusion Matrix
cm_knn <- table(predicted = knn_model, actual = y[ind == 2])

# Model Evaluation
confusionMatrix(cm_knn)

# Calculate accuracy from confusion matrix for KNN
accuracy <- confusionMatrix(cm_knn)$overall['Accuracy']

# Print accuracy for KNN
print(accuracy)

# Calculate precision for each class
precision <- calculate_precision(cm_knn)

# Print precision for each class
print(precision)

# Function to calculate precision for each class
calculate_precision <- function(cm) {
  precisions <- numeric(nrow(cm))
  for (i in 1:nrow(cm)) {
    tp <- cm[i, i]  # True Positives
    fp <- sum(cm[, i]) - tp  # False Positives
    precisions[i] <- tp / (tp + fp)  # Precision
  }
  return(precisions)
}

# Define the function to calculate recall for each class
calculate_recall <- function(cm) {
  recalls <- numeric(nrow(cm))
  for (i in 1:nrow(cm)) {
    tp <- cm[i, i]  # True Positives
    fn <- sum(cm[i, ]) - tp  # False Negatives
    recalls[i] <- tp / (tp + fn)  # Recall
  }
  return(recalls)
}

# Define the function to calculate F1 score for each class
calculate_f1_score <- function(cm) {
  f1_scores <- numeric(nrow(cm))
  for (i in 1:nrow(cm)) {
    tp <- cm[i, i]  # True Positives
    fp <- sum(cm[, i]) - tp  # False Positives
    fn <- sum(cm[i, ]) - tp  # False Negatives
    precision <- tp / (tp + fp)  # Precision
    recall <- tp / (tp + fn)  # Recall
    f1_scores[i] <- 2 * precision * recall / (precision + recall)  # F1 score
  }
  return(f1_scores)
}

# Calculate recall for each class
recall <- calculate_recall(cm_knn)

# Print recall for each class
print(recall)

# Calculate F1 score for each class
f1_score <- calculate_f1_score(cm_knn)

# Print F1 score for each class
print(f1_score)

#Gender: 2, Age: 28, Height: 1.8, Weight:40, family_history_with_overweight: Yes, FAVC: Yes, FCVC: 3, NCP: 2, CAEC:2, SMOKE:0,CH2O: 1
# SCC: 0, FAF: 3, TUE: 1, CALC: 2, MTRANS: 2

# Create a new data frame with new numeric values for Prediction
new_data <- data.frame(
  Gender= c(2),
  Age=c(28),
  Height=c(1.8),
  Weight=c(140),
  family_history_with_overweight=c(1),
  FAVC=c(1),  
  FCVC=c(1),  
  NCP = c(3),  
  CAEC = c(1),
  SMOKE = c(1),
  CH2O = c(1),
  SCC = c(0),
  FAF = c(1),
  TUE = c(0),
  CALC = c(2),  # How often do you drink alcohol?
  MTRANS=c(2),
  NObeyesdad=c(0)
)

predictions <- knn(train = train, test = new_data, cl = y[ind == 1], k = k)

if (predictions == 1) {
  print('New Person is Insufficient_Weight')
} else if (predictions == 2) {
  print('New Person is Normal Weight')
} else if (predictions == 3) {
  print('New Person is Overweight_Level_I')
} else if (predictions == 4) {
  print('New Person is Overweight_Level_II')
} else if (predictions == 5) {
  print('New Person is Obesity_Type_I')
} else if (predictions == 6) {
  print('New Person is Obesity_Type_II')
} else {
  print('New Person is Obesity_Type_III')
}

##################### K means Cluster 

# Select relevant columns for clustering
selected_columns <- c('NCP','FAVC', 'FCVC', 'CALC')

# Filter the dataset to include only selected columns
data_for_clustering <- obesity_data[selected_columns]

head(data_for_clustering)

# Normalize the data
normalized_data <- scale(data_for_clustering)

# Specify the number of clusters
num_clusters <- 7

# Perform k-means clustering
kmeans_result <- kmeans(normalized_data, centers = num_clusters, iter.max = 10, nstart = 3)

# Print cluster assignment for each data point
print(kmeans_result)

fviz_nbclust(normalized_data, kmeans, method = "wss")

# Add cluster assignments to the original dataset
obesity_data$Cluster <- kmeans_result$cluster

head(obesity_data)

# Load required libraries
library(ggplot2)

# Plot results of final k-means clustering
ggplot(obesity_data, aes(x = Age, y = Weight, color = factor(Cluster))) +
  geom_point() +
  labs(title = "K-means Clustering Results",
       x = "Age",
       y = "Weight",
       color = "Cluster") +
  theme_minimal()

# Plot results of final k-means clustering with customized settings
fviz_cluster(kmeans_result, data = normalized_data, geom = "point", # Use points to represent data points
             ellipse.type = "convex", # Use convex hulls for clusters
             ellipse.level = 0.95, # Set the level for ellipse coverage
             palette = "jco", # Color palette for clusters
             ggtheme = theme_minimal() # Apply minimal theme to the plot
)

# Calculate the mean of each variable within each cluster
cluster_means <- aggregate(normalized_data, by = list(cluster = kmeans_result$cluster), FUN = mean)

# Print the cluster means
print(cluster_means)

# Create a new data frame with new numeric values
new_data <- data.frame(
  NCP = c(3),  # How many main meals do you have daily?
  FAVC = c(1),  # Do you eat high caloric food frequently?
  FCVC = c(0),  # Do you usually eat vegetables in your meals?
  CALC = c(1)  # How often do you drink alcohol?
)

# Print the new data
print(new_data)
print(kmeans_result$centers)

# Assuming 'new_data' contains the new numeric data points you want to assign to clusters

# Normalize the new data using the same scaling parameters as the original data
normalized_new_data <- scale(new_data, center = attr(normalized_data, "scaled:center"), scale = attr(normalized_data, "scaled:scale"))

# Calculate Euclidean distances between new data points and cluster centroids
distances <- as.matrix(dist(rbind(kmeans_result$centers, normalized_new_data), method = "euclidean")[1:nrow(kmeans_result$centers)])

# Find the cluster index with the minimum distance for each new data point
predicted_clusters <- apply(distances, 2, which.min)

print("Cluster Assigned to New User: ")
print(predicted_clusters)

##############Prescriptive analysis##########

#Recommendations
# Based on the characteristics of each cluster, provide recommendations tailored to the needs of individuals in each cluster

recommendations_cluster_1 <- "Limit number of main meals to moderate, reduced consumption of high calorie food, frequently consume vegetables."
recommendations_cluster_2 <- "Consume 2 main meals a day, reduce high calorie intake, reduce alcohol intake"
recommendations_cluster_3 <- "Lower the number of main meals in a day, high calorie intake food and alcohol."
recommendations_cluster_4 <- "Lower the number of meals in a day,eat vegetables and consume less alcohol."
recommendations_cluster_5 <- "Eat a balanced 2 main meals a day considering calorie rich food and include vegetables and can consume alcohol moderately"
recommendations_cluster_6 <- "Moderate the number of meals in a day,reducing calories and increasing vegetable intake as well as lowering alcohol consumption."
recommendations_cluster_7 <- "Lower the frequency of meals in a day,lower the high calorie intake food and not consume alcohol"

cluster_number <- 1

if (cluster_number == 1) {
  recommendation <- recommendations_cluster_1
} else if (cluster_number == 2) {
  recommendation <- recommendations_cluster_2
} else if (cluster_number == 3) {
  recommendation <- recommendations_cluster_3
} else if (cluster_number == 4) {
  recommendation <- recommendations_cluster_4
} else if (cluster_number == 5) {
  recommendation <- recommendations_cluster_5
} else if (cluster_number == 6) {
  recommendation <- recommendations_cluster_6
} else if (cluster_number == 7) {
  recommendation <- recommendations_cluster_7
} else {
  recommendation <- "Cluster number not recognized. Please provide a valid cluster number."
}

print("Diet Recommendation for New User: ")
print(recommendation)
