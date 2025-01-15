library(tidytext)
library(text2vec)
library(devtools)
library(e1071)
library(dplyr)
library(caret)
library(tm) # 4.3.3
library(glmnet)
library(textstem) # for stemming and lemmatizing
library(stringr)  
library(gmodels)
library(magrittr)
library(FSelector)


df <- read.csv('MN-DS-news-classification.csv', stringsAsFactors = FALSE) # read csv
new_df <- df %>% filter(category_level_1 == "conflict, war and peace") # doing sub category prediction for "conflict, war and peace"
new_df <- new_df %>% mutate(id = row_number())
new_df$category_level_2 <- factor(new_df$category_level_2)

set.seed(42)

################################################################################################################### necessary function

# function for extracting category-specific relevant terms from the content
relevant_dtm <- function(N, new_df){
  relevant_terms <- new_df %>%
    group_by(category_level_2) %>%
    unnest_tokens(word, content) %>%
    anti_join(stop_words) %>%  # Remove stop words
    count(category_level_2, word, sort = TRUE) %>%
    group_by(category_level_2) %>%
    top_n(N, n) %>%  # Take top N most frequent terms for each category 
    ungroup() %>%
    select(category_level_2, word) # keep only the category and relevant terms
  
  relevant_terms_list <- relevant_terms %>% # create a list for all the relevant terms for each category
    group_by(category_level_2) %>%
    summarise(relevant_words = list(word)) %>%
    ungroup()
  
  clean_content <- function(content, category, relevant_terms_list) { # for each specified category, extract the relevant words
    relevant_words <- relevant_terms_list %>%
      filter(category_level_2 == category) %>%
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
    mutate(content = clean_content(content, category_level_2, relevant_terms_list)) %>%
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
################################################################################################################### performing subcategory prediction 
# SVM model

new_tfidf_dtm <- relevant_dtm(500, new_df)
index <- createDataPartition(new_df$category_level_2, p = 0.70, list = FALSE)
dtm_train <- new_tfidf_dtm[index, ] 
dtm_test <- new_tfidf_dtm[-index, ] 
train <- new_tfidf_general(dtm_train, dtm_test)$train
test <- new_tfidf_general(dtm_train, dtm_test)$test
train_labs <- new_df[index, ]$category_level_2 # 70% of the data for labels
test_labs <- new_df[-index, ]$category_level_2 # 30% of the data for labels

svm_model <- svm(train, train_labs, kernel = "linear")
test_pred <- predict(svm_model, test)
table <- table(Predicted = test_pred, Actual = test_labs)
TP <- diag(table)
accuracy_val <- sum(TP) / sum(table)
cat("Accuracy:", paste0(round(accuracy_val*100, 2), "%"), "\n") # Accuracy:  # Accuracy: 79.58%

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

################################################################################################################### performing subcategory prediction with AFINN sentiment scores 
# SVM model

sentiment_df <- read.csv('final_output.csv', stringsAsFactors = FALSE)
new_sentiment_df <- sentiment_df %>% filter(category_level_1 == "conflict, war and peace")
sentiment_scores <- scale(data.frame(new_sentiment_df$afinn_score))
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
cat("Accuracy:", paste0(round(accuracy*100, 2), "%"), "\n") #Accuracy: 96.25%

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

################################################################################################################### compare with "human interest"
# SVM model

new_df <- df %>% filter(category_level_1 == "human interest")
new_df <- new_df %>% mutate(id = row_number())
new_df$category_level_2 <- factor(new_df$category_level_2)

new_tfidf_dtm <- relevant_dtm(500, new_df)
index <- createDataPartition(new_df$category_level_2, p = 0.70, list = FALSE)
dtm_train <- new_tfidf_dtm[index, ] 
dtm_test <- new_tfidf_dtm[-index, ] 
train <- new_tfidf_general(dtm_train, dtm_test)$train
test <- new_tfidf_general(dtm_train, dtm_test)$test
train_labs <- new_df[index, ]$category_level_2 # 70% of the data for labels
test_labs <- new_df[-index, ]$category_level_2 # 30% of the data for labels

svm_model <- svm(train, train_labs, kernel = "linear")
test_pred <- predict(svm_model, test)
table <- table(Predicted = test_pred, Actual = test_labs)
TP <- diag(table)
accuracy <- sum(TP) / sum(table)
cat("Accuracy:", paste0(round(accuracy*100, 2), "%"), "\n") # Accuracy: 75.56% 

################### performing subcategory prediction with AFINN sentiment scores 
# SVM model

df <- sentiment_df %>% filter(category_level_1 == "human interest")
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
cat("Accuracy:", paste0(round(accuracy*100, 2), "%"), "\n") # Accuracy: 91.67% 

