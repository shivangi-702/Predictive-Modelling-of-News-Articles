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
library(scales)
library(syuzhet)
library(sentimentr)
library(RColorBrewer)
library(wordcloud)
library(ggplot2)
library(lubridate)  # For working with dates

################################################################################################################### getting sentiments of the words 

df <- read.csv('MN-DS-news-classification.csv', stringsAsFactors = FALSE) # read csv
str(df)
df <- df %>% mutate(id = row_number())

################################################################################ getting sentiments of the sentences 

content_senti <- df %>% get_sentences() %>% sentiment_by(by=c('id')) # sentiment score at sentence level with grouping by id
contents <- cbind(df, content_senti) # merging the sentiment scores with the content

content_sentiment <- aggregate(ave_sentiment ~ category_level_1, data = contents, FUN = mean) # aggregate sentiment scores by category_level_1

################################################################################ adding afinn, bing and nrc sentiment values of each entry into the dataframe

afinn_scores <- df %>%
  mutate(row_id = row_number()) %>% # add a unique identifier for each row              
  unnest_tokens(word, content) %>% # break content column into tokens
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(row_id) %>% # group by row identifier
  summarize(afinn_score = sum(value), .groups = "drop") # summaries sentiment score for each row by summing afinn values 

bing_scores <- df %>%
  mutate(row_id = row_number()) %>%               
  unnest_tokens(word, content) %>%
  inner_join(get_sentiments("bing"), by = "word", relationship = "many-to-many") %>% # handle many-to-many for bing lexicon
  group_by(row_id, sentiment) %>% # group by row_id and sentiment type (pos/neg)
  summarize(count = n(), .groups = "drop") %>% # count occurence for each sentiment
  pivot_wider(names_from = sentiment, values_from = count, values_fill = 0) %>%
  mutate(bing_sentiment = case_when( # overall sentiment based on counts
    positive > negative ~ "positive",
    negative > positive ~ "negative",
    TRUE ~ "neutral"
  )) %>%
  select(row_id, bing_sentiment)

nrc_scores <- df %>%
  mutate(row_id = row_number()) %>%                
  unnest_tokens(word, content) %>%
  inner_join(get_sentiments("nrc"), by = "word", relationship = "many-to-many") %>% 
  group_by(row_id, sentiment) %>%
  summarize(count = n(), .groups = "drop") %>%
  group_by(row_id) %>%
  filter(count == max(count)) %>% # keep the dominant sentiment for each row
  slice(1) %>% # breaks ties by taking the first occurrence
  select(row_id, nrc_sentiment = sentiment)

final_output <- df %>%
  mutate(row_id = row_number()) %>% # merge based on row_id
  left_join(afinn_scores, by = "row_id") %>%
  left_join(bing_scores, by = "row_id") %>%
  left_join(nrc_scores, by = "row_id") %>%
  select(-row_id) %>% # remove row_id after joining
  mutate(
    afinn_score = ifelse(is.na(afinn_score), 0, afinn_score),        # replace NA in afinn_score with 0
    bing_sentiment = ifelse(is.na(bing_sentiment), "neutral", bing_sentiment),  # replace NA in bing_sentiment with "neutral"
    nrc_sentiment = ifelse(is.na(nrc_sentiment), "neutral", nrc_sentiment)   # replace NA in nrc_sentiment with "neutral"
  )

head(final_output,1)

################################################################################################################### creating the different visualisations

# bar plot of the sentiment scores for each category_level_1
ggplot(data = content_sentiment, aes(x = category_level_1, y = ave_sentiment, fill = category_level_1)) + 
  geom_bar(stat = "identity") +                               
  labs(title = "Sentiment Analysis of News Content",          
       x = "Category Level 1", y = "Sentiment Score") +      
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) 

# box plot of category_level_1 distribution in afinn sentiment score
ggplot(final_output, aes(x = category_level_1, y = afinn_score, fill = category_level_1)) +
  geom_boxplot() +
  labs(title = "AFINN Sentiment Score by News Category", x = "Category Level 1", y = "AFINN Score") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# bar plot displaying extremely positive, extremely negative and neutral counts for each category_level_1
final_output %>%
  mutate(sentiment_type = case_when(
    afinn_score >= 10 ~ "extremely positive",
    afinn_score <= -10 ~ "extremely negative",
    TRUE ~ "neutral"
  )) %>%
  group_by(category_level_1, sentiment_type) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = category_level_1, y = count, fill = sentiment_type)) +
  geom_col(position = "dodge") +
  labs(title = "Extreme Sentiments by Category", x = "Category Level 1", y = "Count") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# heatmap of sentiments by category level and counts
final_output %>%
  count(category_level_1, nrc_sentiment) %>%
  ggplot(aes(x = category_level_1, y = nrc_sentiment, fill = n)) +
  geom_tile() +
  scale_fill_gradient(low = "orange", high = "blue") +
  labs(title = "Heatmap of Sentiments by Category",
       x = "Category Level 1",
       y = "Sentiment",
       fill = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

################################################################################ selecting which category_level_1 to do predictive modelling for

# filter category_level_1 categories with both high frequency and extreme sentiments
important_combined <- final_output %>%
  group_by(category_level_1) %>%
  summarize(
    avg_afinn = mean(afinn_score, na.rm = TRUE),
    count = n()
  ) %>%
  filter(count > 50 & abs(avg_afinn) > 20)  # Frequent with strong sentiment
categories <- final_output %>%
  filter(category_level_1 %in% important_combined$category_level_1)
unique(categories$category_level_1) # "crime, law and justice", "human interest", "lifestyle and leisure", "conflict, war and peace"

categories %>%
  group_by(category_level_1) %>%
  summarize(
    num_subcategories = n_distinct(category_level_2),
    total_entries = n()
  ) %>%
  arrange(desc(num_subcategories)) # understanding how many sub-categories and total entries are there

categories %>%
  group_by(category_level_1, category_level_2) %>%
  summarize(count = n(), .groups = "drop") %>%
  group_by(category_level_1) %>%
  summarize(
    max_subcategory_ratio = max(count) / sum(count),
    total_entries = sum(count)
  ) %>%
  arrange(max_subcategory_ratio) # understanding max sub-category ratios and total entries are there
# lifestyle and leisure not ideal because high dominance of one subcategory and insufficient data

# bar plot to visualize the different sentiments of each subcategory for each category 
categories %>%
  group_by(category_level_1, category_level_2) %>%
  summarize(avg_afinn = mean(afinn_score, na.rm = TRUE), .groups = "drop") %>%
  ggplot(aes(x = category_level_2, y = avg_afinn, fill = category_level_1)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Average Sentiment by Subcategory and Main Category",
    x = "Subcategory",
    y = "Average AFINN Score"
  ) +
  theme_minimal() 
# ideal category for predictive modelling: conflict, war and peace 

write.csv(final_output, "final_output.csv", row.names = FALSE)
final_output <- read.csv("~/Desktop/Intro to Data Science/final_output.csv") # save dataframe with sentiment values

