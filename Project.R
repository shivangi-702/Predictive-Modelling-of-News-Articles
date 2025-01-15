library(tidytext)
library(text2vec)
library(devtools)
library(e1071)
library(dplyr)
library(caret)
library(glmnet)
library(textstem) # for stemming and lemmatizing
library(stringr)  
library(gmodels)
library(magrittr)
library(FSelector)
library(randomForest)
library(FactoMineR)
library(tidyr)
library(tm)


new_df <- read.csv('MN-DS-news-classification.csv', stringsAsFactors = FALSE) # read csv
str(new_df)
head(new_df,1)
colSums(is.na(new_df)) # checking NA values 

table(new_df$category_level_1) # checking different primary/main categories
table(new_df$category_level_2) # checking different sub categories
barplot(
  table(new_df$category_level_1), 
  col = "purple", 
  ylab = "Total Number", 
  main = "Category",
  las = 2
)

length(new_df$content)
set.seed(42) # setting random seed for reproducibility
new_df <- new_df[sample(nrow(new_df)), ]

###################################################################################################################

new_df$category_level_1 <- factor(new_df$category_level_1) # changing dependent variable into a factor
str(new_df$category_level_1) # 17 categories
table(new_df$category_level_1) # checking counts of each main category

corpus <- VCorpus(VectorSource(new_df$content)) # creating a corpus of the content
print(corpus) # 10917 documents
as.character(corpus[[1]]) # examine an actual line of text in corpus

corpus_clean <- tm_map(corpus, content_transformer(tolower)) #lowercase
corpus_clean <- tm_map(corpus_clean, removeNumbers) #remove numbers 
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords()) # remove stop words
corpus_clean <- tm_map(corpus_clean, removePunctuation) # remove punctuation
corpus_clean <- tm_map(corpus_clean, stripWhitespace) #remove white spaces
corpus_clean <- tm_map(corpus_clean, stemDocument) #stem the words
as.character(corpus_clean[[1]])

dtm <- DocumentTermMatrix(corpus_clean)

index <- createDataPartition(new_df$category_level_1, p = 0.70, list = FALSE)
dtm_train <- dtm[index, ] # 70% of the data
dtm_test <- dtm[-index, ] # 30% of the data
train_labs <- new_df[index, ]$category_level_1 # 70% of the data for labels
test_labs <- new_df[-index, ]$category_level_1 # 30% of the data for labels

# check if a word is present in the line and assign yes or on in the matrix
convert_counts <- function(x){
  x <- ifelse(x>0, "Yes", "No")
}

# train, test split function
general <- function(dtm_train, dtm_test){
  freq_words <- findFreqTerms(dtm_train, 5) # finding terms that appear 5 or more times
  # include only frequent terms in the training and test data
  dtm_freq_train <- dtm_train[ , freq_words]
  dtm_freq_test <- dtm_test[ , freq_words]
  train <- apply(dtm_freq_train, MARGIN = 2, convert_counts) # for training
  test <- apply(dtm_freq_test, MARGIN = 2, convert_counts) # for testing
  return(list(train = train, test = test))
}

# train, test split function for tfidf dtm
tfidf_general <- function(dtm_train, dtm_test){
  freq_words <- findFreqTerms(dtm_train, 5) # finding terms that appear 5 or more times
  # include only frequent terms in the training and test data
  dtm_freq_train <- dtm_train[ , freq_words]
  dtm_freq_test <- dtm_test[ , freq_words]
  train <- data.matrix(dtm_freq_train)
  test <- data.matrix(dtm_freq_test)
  return(list(train = train, test = test))
}

# acuuracy function
accuracy_func <- function(test_pred, test_labs){
  table <- table(Predicted = test_pred, Actual = test_labs)
  TP <- diag(table)
  accuracy_val <- sum(TP) / sum(table)
  return(accuracy_val)
}

################################################################################################################### using sparse dtm with Naive Bayes
# NaiveBayes model

train <- general(dtm_train, dtm_test)$train
test <- general(dtm_train, dtm_test)$test
classifier <- naiveBayes(train, train_labs)
test_pred <- predict(classifier, test)
cat("Accuracy:", paste0(round(accuracy_func(test_pred, test_labs)*100, 2), "%"), "\n") # Accuracy: 57.79% 

################################################################################################################### using non-sparse dtm with Naive Bayes

sparse_dtm <- removeSparseTerms(dtm, 0.98) # terms with sparsity above 98% removed from document-term matrix
dtm_train <- sparse_dtm[index, ] # 70% of the data
dtm_test <- sparse_dtm[-index, ] # 30% of the data
train <- general(dtm_train, dtm_test)$train
test <- general(dtm_train, dtm_test)$test
classifier <- naiveBayes(train, train_labs)
test_pred <- predict(classifier, test)
cat("Accuracy:", paste0(round(accuracy_func(test_pred, test_labs)*100, 2), "%"), "\n") # Accuracy: 52.58% 

################################################################################################################### using tfidf dtm with Naive Bayes

tfidf_dtm <- DocumentTermMatrix(corpus_clean, control = list(weighting = weightTfIdf))
dtm_train <- tfidf_dtm[index, ] # 70% of the data
dtm_test <- tfidf_dtm[-index, ] # 30% of the data
train <- tfidf_general(dtm_train, dtm_test)$train
test <- tfidf_general(dtm_train, dtm_test)$test
classifier <- naiveBayes(train, train_labs)
test_pred <- predict(classifier, test)
cat("Accuracy:", paste0(round(accuracy_func(test_pred, test_labs)*100, 2), "%"), "\n") # Accuracy: 51.92% 

################################################################################################################### using tfidf dtm with RandomForest
# RandomForest model

rf_model <- randomForest(train, train_labs)
test_pred <- predict(rf_model, test)
cat("Accuracy:", paste0(round(accuracy_func(test_pred, test_labs)*100, 2), "%"), "\n") # Accuracy: 69.36% 

################################################################################################################### using tfidf dtm with SVM
# SVM model

svm_model <- svm(train, train_labs, kernel = "linear")
test_pred <- predict(svm_model, test)
cat("Accuracy:", paste0(round(accuracy_func(test_pred, test_labs)*100, 2), "%"), "\n") # Accuracy: 71.01% 

################################################################################################################### Filtering the tfidf dtm and re-try with RandomForest
# SVM model

selected_terms <- findFreqTerms(tfidf_dtm, lowfreq = 10, highfreq = 0.8 * nrow(tfidf_dtm)) # terms with frequency between 10 and 80% of the total documents were selected
length(selected_terms) # removed 1085 terms from 78370
new_tfidf_dtm <- tfidf_dtm[, selected_terms] # filter the tfidf docuemnt term matrix to have only the selected terms
dtm_train <- new_tfidf_dtm[index, ] 
dtm_test <- new_tfidf_dtm[-index, ] 

train <- tfidf_general(dtm_train, dtm_test)$train
test <- tfidf_general(dtm_train, dtm_test)$test
svm_model <- svm(train, train_labs)
test_pred <- predict(svm_model, test)
cat("Accuracy:", paste0(round(accuracy_func(test_pred, test_labs)*100, 2), "%"), "\n") # Accuracy: 69.18% 

################################################################################################################### Remove terms that don't contribute to predicting the label using Mutual Information
# SVM model

tfidf_dtm_new <- removeSparseTerms(tfidf_dtm, 0.95) # terms with sparsity above 95% removed from document-term matrix
tfidf_df <- as.data.frame(as.matrix(tfidf_dtm_new)) # convert tfidf document term matrix to data frame
label <- new_df$category_level_1 
mi <- information.gain(label ~ ., data = tfidf_df) # calculate information gain for each terms relative to the labels
selected_terms <- rownames(mi) # only select terms with high information gain
selected_terms

tfidf_filtered_dtm <- tfidf_dtm[, selected_terms]
dtm_train <- tfidf_filtered_dtm[index, ] 
dtm_test <- tfidf_filtered_dtm[-index, ] 
train <- tfidf_general(dtm_train, dtm_test)$train
test <- tfidf_general(dtm_train, dtm_test)$test
svm_model <- svm(train, train_labs)
test_pred <- predict(svm_model, test)
cat("Accuracy:", paste0(round(accuracy_func(test_pred, test_labs)*100, 2), "%"), "\n") # Accuracy: 65.97%

################################################################################################################### trying PCA to reduce dimentionality
# SVM model

tfidf_dtm_new <- removeSparseTerms(tfidf_dtm, 0.95) # terms with sparsity above 95% removed from document-term matrix
term_freqs <- colSums(as.matrix(tfidf_dtm_new)) # sum frequencies for each term
freq_words <- names(term_freqs[term_freqs >= 5]) # select terms with frequency >= 5
tfidf_dtm_new <- tfidf_dtm_new[ , freq_words] # subset dtm with only frequent terms 
ncol(tfidf_dtm) # dimensions 127668
pca <- PCA(as.matrix(tfidf_dtm_new), ncp = 500)  # reduce to 500 dimensions
reduced_features <- pca$ind$coord
length(reduced_features)
labels <- new_df$category_level_1
data <- as.data.frame(reduced_features)
data$label <- labels
train_data <- data[index, ]
test_data <- data[-index, ]

train <- train_data[, -ncol(train_data)]  # All columns except the last (label)
train_labels_new <- train_data$label
test <- test_data[, -ncol(test_data)]
test_labels_new <- test_data$label
svm_model <- svm(train, train_labs)
test_pred <- predict(svm_model, test)
cat("Accuracy:", paste0(round(accuracy_func(test_pred, test_labs)*100, 2), "%"), "\n") # Accuracy: 68.63% 62.13%

################################################################################################################### looking at category-specific relevant content terms 

# function for extracting category-specific relevant terms from the content
relevant_dtm <- function(N, new_df){
  relevant_terms <- new_df %>%
    group_by(category_level_1) %>%
    unnest_tokens(word, content) %>%
    anti_join(stop_words) %>%  # Remove stop words
    count(category_level_1, word, sort = TRUE) %>%
    group_by(category_level_1) %>%
    top_n(N, n) %>%  # Take top N most frequent terms for each category 
    ungroup() %>%
    select(category_level_1, word) # keep only the category and relevant terms
  
  relevant_terms_list <- relevant_terms %>% # create a list for all the relevant terms for each category
    group_by(category_level_1) %>%
    summarise(relevant_words = list(word)) %>%
    ungroup()
  
  clean_content <- function(content, category, relevant_terms_list) { # for each specified category, extract the relevant words
    relevant_words <- relevant_terms_list %>%
      filter(category_level_1 == category) %>%
      pull(relevant_words) %>% 
      unlist()
    
    content <- tolower(content) # make the content lowercase
    relevant_words <- tolower(relevant_words) # make the relevant terms lowercase
    words <- unlist(strsplit(content, "\\s+")) # split the content into single words
    cleaned_words <- words[words %in% relevant_words] # remove non-relevant terms 
    return(paste(cleaned_words, collapse = " ")) # return cleaned content as a string
  }
  
  temp <- new_df %>% # create a new data frame using the new cleaned content
    rowwise() %>%
    mutate(content = clean_content(content, category_level_1, relevant_terms_list)) %>%
    ungroup()
  
  # clean the new data frame with only relevant terms 
  new_corpus <- VCorpus(VectorSource(temp$content)) # 10917 documents
  new_corpus_clean <- tm_map(new_corpus, content_transformer(tolower))
  new_corpus_clean <- tm_map(new_corpus_clean, removeNumbers)
  new_corpus_clean <- tm_map(new_corpus_clean, removeWords, stopwords())
  new_corpus_clean <- tm_map(new_corpus_clean, removePunctuation)
  new_corpus_clean <- tm_map(new_corpus_clean, stripWhitespace)
  new_corpus_clean <- tm_map(new_corpus_clean, stemDocument)
  
  new_tfidf_dtm <- DocumentTermMatrix(new_corpus_clean, control = list(weighting = weightTfIdf))
  return(new_tfidf_dtm)
}

# function for creating train and test
new_tfidf_general <- function(dtm_train, dtm_test){
  freq_words <- findFreqTerms(dtm_train, 5) # finding terms that appear 5 or more times
  # include only frequent terms in the training and test data
  dtm_freq_train <- dtm_train[ , freq_words]
  dtm_freq_test <- dtm_test[ , freq_words]
  train <- data.matrix(dtm_freq_train)
  test <- data.matrix(dtm_freq_test)
  
  constant_vars <- nearZeroVar(train, saveMetrics = TRUE) # find variables with an almost zero variance
  train <- train[, !constant_vars$zeroVar] # remove these constant variables 
  test <- test[, colnames(train), drop = FALSE] # making sure test has the same columns as train
  
  return(list(train = train, test = test))
}

################################################################################################################### testing all 3 models with 1000 category-specific relevant content terms 

new_tfidf_dtm <- relevant_dtm(1000, new_df)
dtm_train <- new_tfidf_dtm[index, ] 
dtm_test <- new_tfidf_dtm[-index, ] 
train <- new_tfidf_general(dtm_train, dtm_test)$train
test <- new_tfidf_general(dtm_train, dtm_test)$test

# trying RandomForest model
rf_model <- randomForest(train, train_labs)
test_pred <- predict(rf_model, test)
cat("Accuracy:", paste0(round(accuracy_func(test_pred, test_labs)*100, 2), "%"), "\n") # Accuracy: 82.74%

### trying SVM model
svm_model <- svm(train, train_labs, kernel = "linear")
test_pred <- predict(svm_model, test)
cat("Accuracy:", paste0(round(accuracy_func(test_pred, test_labs)*100, 2), "%"), "\n") # Accuracy: 93.01% <- best performing model

# try NaiveBayes model
train <- apply(dtm_train, MARGIN = 2, convert_counts)
test <- apply(dtm_test, MARGIN = 2, convert_counts)
constant_vars <- nearZeroVar(train, saveMetrics = TRUE)
train <- train[, !constant_vars$zeroVar]
test <- test[, colnames(train), drop = FALSE]
classifier <- naiveBayes(train, train_labs)
test_pred <- predict(classifier, test)
cat("Accuracy:", paste0(round(accuracy_func(test_pred, test_labs)*100, 2), "%"), "\n") # Accuracy: 92.67%

################################################################################################################### 1500 category-specific relevant content terms 
# SVM model

new_tfidf_dtm <- relevant_dtm(1500, new_df)
dtm_train <- new_tfidf_dtm[index, ] 
dtm_test <- new_tfidf_dtm[-index, ] 
train <- new_tfidf_general(dtm_train, dtm_test)$train
test <- new_tfidf_general(dtm_train, dtm_test)$test

svm_model <- svm(train, train_labs, kernel = "linear")
test_pred <- predict(svm_model, test)
cat("Accuracy:", paste0(round(accuracy_func(test_pred, test_labs)*100, 2), "%"), "\n") # Accuracy: 88.73% 

################################################################################################################### 500 category-specific relevant content terms 
# SVM model

new_tfidf_dtm <- relevant_dtm(500, new_df)
dtm_train <- new_tfidf_dtm[index, ] 
dtm_test <- new_tfidf_dtm[-index, ] 
train <- new_tfidf_general(dtm_train, dtm_test)$train
test <- new_tfidf_general(dtm_train, dtm_test)$test

svm_model <- svm(train, train_labs, kernel = "linear")
test_pred <- predict(svm_model, test)
table <- table(Predicted = test_pred, Actual = test_labs)
TP <- diag(table)
accuracy_val <- sum(TP) / sum(table)
cat("Accuracy:", paste0(round(accuracy_val*100, 2), "%"), "\n") # Accuracy: 95.91% <- best accuracy

FP <- colSums(table) - TP
FN <- rowSums(table) - TP
TN <- sum(table) - (FP + FN + TP)

precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1 <- 2 * (precision * recall) / (precision + recall)

result <- data.frame( # dataframe for each main category and its F1 score
  Category = names(f1),  
  F1_Score = round(f1 * 100, 2)# F1 scores as percentages
)
rownames(result) <- NULL
print(result[order(result$F1_Score, decreasing = TRUE), ])

###################################################################################################################
# Trying to use title as input text instead of content
################################################################################################################### 

# modified function for extracting category-specific relevant terms: using title instead of content
relevant_dtm_title <- function(N, new_df){ 
  relevant_terms <- new_df %>%
    group_by(category_level_1) %>%
    unnest_tokens(word, title) %>%
    anti_join(stop_words) %>%  # Remove stop words
    count(category_level_1, word, sort = TRUE) %>%
    group_by(category_level_1) %>%
    top_n(N, n) %>%  # Take top N most frequent terms for each category 
    ungroup() %>%
    select(category_level_1, word) # keep only the category and relevant terms
  
  relevant_terms_list <- relevant_terms %>% # create a list for all the relevant terms for each category
    group_by(category_level_1) %>%
    summarise(relevant_words = list(word)) %>%
    ungroup()
  
  clean_content <- function(title, category, relevant_terms_list) { # for each specified category, extract the relevant words
    relevant_words <- relevant_terms_list %>%
      filter(category_level_1 == category) %>%
      pull(relevant_words) %>% 
      unlist()
    
    title <- tolower(title) # make the title lowercase
    relevant_words <- tolower(relevant_words) # make the relevant terms lowercase
    words <- unlist(strsplit(title, "\\s+")) # split the title into single words
    cleaned_words <- words[words %in% relevant_words] # remove non-relevant terms 
    return(paste(cleaned_words, collapse = " ")) # return cleaned title as a string
  }
  
  temp <- new_df %>% # create a new data frame using the new cleaned title
    rowwise() %>%
    mutate(title = clean_content(title, category_level_1, relevant_terms_list)) %>%
    ungroup()
  
  # clean the new data frame with only relevant terms 
  new_corpus <- VCorpus(VectorSource(temp$title)) # 10917 documents
  new_corpus_clean <- tm_map(new_corpus, content_transformer(tolower))
  new_corpus_clean <- tm_map(new_corpus_clean, removeNumbers)
  new_corpus_clean <- tm_map(new_corpus_clean, removeWords, stopwords())
  new_corpus_clean <- tm_map(new_corpus_clean, removePunctuation)
  new_corpus_clean <- tm_map(new_corpus_clean, stripWhitespace)
  new_corpus_clean <- tm_map(new_corpus_clean, stemDocument)
  
  new_tfidf_dtm <- DocumentTermMatrix(new_corpus_clean, control = list(weighting = weightTfIdf))
  return(new_tfidf_dtm)
}

################################################################################################################### 
# SVM model

# trying with 100 category-specific relevant title terms
new_tfidf_dtm <- relevant_dtm_title(100, new_df)
dtm_train <- new_tfidf_dtm[index, ] 
dtm_test <- new_tfidf_dtm[-index, ] 
train <- new_tfidf_general(dtm_train, dtm_test)$train
test <- new_tfidf_general(dtm_train, dtm_test)$test
svm_model <- svm(train, train_labs, kernel = "linear")
test_pred <- predict(svm_model, test)
cat("Accuracy:", paste0(round(accuracy_func(test_pred, test_labs)*100, 2), "%"), "\n") # Accuracy: 82.16.39% 

# trying with 50 category-specific relevant title terms
new_tfidf_dtm <- relevant_dtm_title(50, new_df)
dtm_train <- new_tfidf_dtm[index, ] 
dtm_test <- new_tfidf_dtm[-index, ] 
train <- new_tfidf_general(dtm_train, dtm_test)$train
test <- new_tfidf_general(dtm_train, dtm_test)$test
svm_model <- svm(train, train_labs, kernel = "linear")
test_pred <- predict(svm_model, test)
cat("Accuracy:", paste0(round(accuracy_func(test_pred, test_labs)*100, 2), "%"), "\n") # Accuracy: 80.39% 

# trying with 150 category-specific relevant title terms
new_tfidf_dtm <- relevant_dtm_title(150, new_df)
dtm_train <- new_tfidf_dtm[index, ] 
dtm_test <- new_tfidf_dtm[-index, ] 
train <- new_tfidf_general(dtm_train, dtm_test)$train
test <- new_tfidf_general(dtm_train, dtm_test)$test
svm_model <- svm(train, train_labs, kernel = "linear")
test_pred <- predict(svm_model, test)
cat("Accuracy:", paste0(round(accuracy_func(test_pred, test_labs)*100, 2), "%"), "\n") # Accuracy: 82.07% 

################################################################################################################### 

# FINAL_OUTPUT USING SENTIMENTS

################################################################################################################### 

df <- read.csv('final_output.csv', stringsAsFactors = FALSE) # read csv
df <- df %>% mutate(id = row_number())
df$category_level_1 <- factor(df$category_level_1)

################################################################################################################### trying with 500 category-specific relevant content terms + sentiments
# SVM model

new_tfidf_dtm <- relevant_dtm(500, new_df)
sentiment_scores <- scale(data.frame(df$afinn_score))
combined_features <- cbind(as.matrix(new_tfidf_dtm), sentiment_scores)
dtm_train <- combined_features[index, ] 
dtm_test <- combined_features[-index, ] 
train <- data.matrix(dtm_train)
test <- data.matrix(dtm_test)

constant_vars <- nearZeroVar(train, saveMetrics = TRUE)
train <- train[, !constant_vars$zeroVar]
test <- test[, colnames(train), drop = FALSE]

svm_model <- svm(train, train_labs, kernel = "linear")
test_pred <- predict(svm_model, test)
table <- table(Predicted = test_pred, Actual = test_labs)
TP <- diag(table)
accuracy <- sum(TP) / sum(table)
cat("Accuracy:", paste0(round(accuracy*100, 2), "%"), "\n") # Accuracy: 94.38% 

################################################################################################################### trying with 100 category-specific relevant title terms + sentiments
# SVM model

new_tfidf_dtm <- relevant_dtm_title(100, new_df)
combined_features <- cbind(as.matrix(new_tfidf_dtm), sentiment_scores)
dtm_train <- combined_features[index, ] 
dtm_test <- combined_features[-index, ] 
train <- data.matrix(dtm_train)
test <- data.matrix(dtm_test)

constant_vars <- nearZeroVar(train, saveMetrics = TRUE)
train <- train[, !constant_vars$zeroVar]
test <- test[, colnames(train), drop = FALSE]

svm_model <- svm(train, train_labs, kernel = "linear")
test_pred <- predict(svm_model, test)
table <- table(Predicted = test_pred, Actual = test_labs)
TP <- diag(table)
accuracy <- sum(TP) / sum(table)
cat("Accuracy:", paste0(round(accuracy*100, 2), "%"), "\n") # Accuracy: 82.07% 

