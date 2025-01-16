# Predictive Modelling of News Articles

This project emphasizes predicting news article categories and subcategories using various machine learning models and feature extraction techniques. The experiments concluded that:

The optimal SVM model performed best when using a TF-IDF document term matrix of the news content with 500 category-specific relevant terms.
Titles as inputs underperformed significantly, even when combined with AFINN sentiment scores.
The addition of sentiment scores improved predictive performance at the subcategory level, suggesting that sentiments enhance predictive power for specific topics or aspects compared to broader texts.
Sentiment analysis is a valuable feature for exploratory data analysis (EDA) and understanding the emotional tone of news categories.

# Files
1. sentiment_analysis_code.r: Contains the code for performing sentiment analysis on the dataset.
2. project.r: Main script that handles data preprocessing, feature extraction, and model training for category prediction.
3. predictive_subcategory.r: Focuses on predicting subcategories of news articles.

# Installation
To run the R code in this project, you need to install the following packages:

install.packages(c("tidytext", "text2vec", "devtools", "e1071", "dplyr", "caret", "glmnet", 
                   "textstem", "stringr", "gmodels", "magrittr", "FSelector", "randomForest", 
                   "FactoMineR", "tidyr", "tm", "scales", "syuzhet", "sentimentr", 
                   "RColorBrewer", "wordcloud", "ggplot2", "lubridate"))
                   
Ensure all these packages are installed before running the scripts to avoid any issues with missing dependencies.

# Usage
1. Start with sentiment_analysis_code.r to perform sentiment analysis on the dataset.
2. Run project.r to preprocess the data, extract features, and train models for category prediction.
3. Execute predictive_subcategory.r to focus on subcategory prediction.
