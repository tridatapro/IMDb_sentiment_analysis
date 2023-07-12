# Unmasking the Hidden Emotions: Sentiment Analysis on IMDb Movie Reviews
## Introduction
In today’s digital era, understanding people’s sentiments and emotions has become a valuable asset. IMDb, the Internet Movie Database, is a platform that hosts a vast collection of movies, TV shows, and user-generated reviews. In this article, we embark on an exciting journey into sentiment analysis on IMDb movie reviews. By leveraging the power of machine learning and Python, we delve deep into the sea of IMDb reviews, aiming to unmask the hidden emotions buried within. Join us as we explore the intricacies of sentiment analysis, from data collection and preprocessing to model training and prediction. Get ready to dive into the captivating world of IMDb movie reviews, where we decipher the sentiments and emotions that shape our cinematic experiences.

## Objective
The objective of this project is to perform sentiment analysis on IMDb movie reviews using machine learning techniques and Python. By applying natural language processing (NLP) and text mining methodologies, we aim to analyze the sentiment expressed in user reviews and classify them as positive or negative. Through this project, we seek to gain insights into the overall sentiment trends and understand the emotional reactions of moviegoers toward different films. Additionally, we aim to develop a trained model that can accurately predict the sentiment of new, unseen movie reviews. By achieving these objectives, we aim to showcase the power of sentiment analysis in understanding and interpreting public opinion in the realm of movies.

## Data Source
The data for this project is obtained from the IMDb Dataset available on Kaggle. The dataset, titled “IMDb Dataset: Sentiment Analysis in CSV Format,” provides a collection of movie reviews along with their corresponding sentiment labels. The dataset can be accessed at the following link: [IMDb Dataset: Sentiment Analysis in CSV Format](https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format?resource=download). It contains a diverse range of movie reviews, allowing us to perform sentiment analysis and gain insights into the sentiments expressed by users. The dataset is in CSV format, making it easily accessible and compatible with popular data analysis tools and libraries in Python.

## Data Cleaning and Preparation
Before performing sentiment analysis on the IMDb movie review dataset, it is essential to perform data cleaning and preparation to ensure the quality and suitability of the data for analysis. The following steps were undertaken:

Step 1: Import the necessary libraries

The required libraries such as pandas, numpy, matplotlib, seaborn, and CountVectorizer are imported to perform various operations.

Step 2: Initialize the CountVectorizer

In sentiment analysis, textual data needs to be transformed into numerical representations that machine learning algorithms can process. The CountVectorizer is a popular tool used for text preprocessing and feature extraction.

To initialize the CountVectorizer, we first import the necessary libraries, including CountVectorizer from the sklearn.feature_extraction.text module. This module provides a range of tools for extracting features from text data.

In this project, we import the CountVectorizer class from sklearn.feature_extraction.text. We then create an instance of the CountVectorizer class and assign it to the variable count.

The CountVectorizer is a flexible tool that allows us to convert a collection of text documents into a matrix representation, where each row represents a document and each column represents a unique word or term from the corpus. It counts the frequency of each word in the document and creates a numerical representation that can be used for further analysis and modeling.

By initializing the CountVectorizer, we set the stage for text preprocessing and feature extraction, which are crucial steps in sentiment analysis.

Step 3: Load the dataset

The dataset is loaded from the “Train.csv” file using the pd.read_csv() function and stored in the 'data' DataFrame.

Step 4: Data Exploration

Before diving into sentiment analysis, it is essential to explore and understand the dataset. In this step, we perform various exploratory operations to gain insights into the data.

First, we check the shape of the dataset using data.shape. This provides information about the number of rows and columns in the dataset, giving us an idea of its size.

Next, we examine the information of the dataset using data.info(). This provides a summary of the dataset's structure, including the column names, data types, and the presence of any missing values. Understanding the data types and missing values helps us in data preprocessing and handling.

To gain a better understanding of the dataset’s statistical properties, we use data.describe(). This generates descriptive statistics such as count, mean, standard deviation, minimum, and maximum values for numerical columns. It gives us insights into the central tendency, dispersion, and range of the data.

Additionally, to visualize the distribution of sentiment labels in the dataset, we create a pie chart using the matplotlib library. This chart helps us visualize the proportion of positive and negative sentiment labels. By using colors, labels, and explode options, we can enhance the visual representation of the sentiment distribution.

Step 5: Text preprocessing

In order to clean the text data and prepare it for sentiment analysis, a preprocessing function called preprocess_text() was defined. This function utilized the ‘re’ library for text manipulation and transformation. The following steps were performed within the function:

1. URL Removal: The function used regular expressions to remove URLs from the text data, ensuring that web addresses did not interfere with the sentiment analysis process.
2. Unwanted Character Removal: Unwanted characters and punctuation were eliminated from the text using regular expressions. This step helped to remove noise and irrelevant symbols that could potentially impact the sentiment analysis results.
3. Text Lowercasing: The text was converted to lowercase to ensure consistency in the analysis. This step addressed the issue of mixed cases that might occur in the original dataset.

Step 6: Download NLTK Resources

To enhance the text preprocessing capabilities, the Natural Language Toolkit (NLTK) was utilized in this sentiment analysis project. The NLTK library provides various resources and tools for text processing and analysis. In this step, the stopwords and punkt tokenizer resources from NLTK were downloaded to facilitate the text preprocessing process.

The following actions were performed in this step:
1. Importing NLTK: The NLTK library was imported to gain access to its resources and functionalities.
2. Downloading Stopwords: Stopwords are commonly used words in a language (e.g., “the”, “is”, “and”) that do not carry significant meaning and can be excluded from the text analysis. By downloading the stopwords resource from NLTK, a predefined set of stopwords for the English language was made available for further use in the sentiment analysis pipeline.
3. Downloading Punkt Tokenizer: The punkt tokenizer is a pre-trained model for tokenizing text into sentences or words. It can accurately segment text into meaningful units. By downloading the punkt tokenizer resource from NLTK, the sentiment analysis model could leverage its capabilities for better text tokenization and preprocessing.

Step 7: Text Preprocessing using NLTK

To further enhance the quality of the text data and prepare it for sentiment analysis, an additional text preprocessing step was implemented using NLTK. This step involved tokenization, removal of punctuation, stopwords, and performing stemming using the PorterStemmer algorithm.

The following actions were performed in this step:

Defining the Preprocessing Function: A function named preprocess_text() was defined to encapsulate the text preprocessing operations. This function takes a piece of text as input and applies the following transformations:
1. Tokenization: The text is tokenized into individual words using the word_tokenize() function from NLTK.
2. Removal of Punctuation: Punctuation marks and special characters are removed from the tokenized words.
3. Removal of Stopwords: NLTK’s predefined set of stopwords for the English language is used to filter out common and insignificant words.
4. Stemming: The PorterStemmer algorithm from NLTK is applied to reduce words to their base or root form, capturing the core meaning of the word.

Applying Preprocessing to the DataFrame: The preprocess_text() function is applied to the 'text' column in the DataFrame using the apply() function. The output of the preprocessing is stored in a new column called 'preprocessed_text', which contains the cleaned and transformed text data ready for further analysis.

By implementing this text preprocessing step, the raw text data is transformed into a cleaner and more standardized format, which improves the effectiveness of the subsequent sentiment analysis tasks. The ‘preprocessed_text’ column now contains the preprocessed text ready for exploration and modeling.

Step 8: Train-Test Split

To evaluate the performance of the sentiment analysis model accurately, it is crucial to divide the dataset into separate training and testing sets. This step ensures that the model is trained on one portion of the data and evaluated on unseen data to assess its generalization ability.

In this step, the preprocessed text data is assigned to ‘X’, representing the input features, and the corresponding labels are assigned to ‘y’, representing the target variable (sentiment labels).

The scikit-learn library provides the train_test_split() function, which facilitates the division of the dataset into training and testing sets. The split is performed by specifying the desired test size (e.g., 0.2 for 20% of the data) and optionally setting a random state for reproducibility.

In the above code, the ‘preprocessed_text’ column is assigned to ‘X’, and the ‘label’ column (containing the sentiment labels) is assigned to ‘y’. The train_test_split() function is called with 'X', 'y', and additional parameters such as the test size (0.2) and random state (42) to ensure consistent splits across different runs.

After executing this step, ‘X_train’ and ‘y_train’ will represent the training data, while ‘X_test’ and ‘y_test’ will contain the testing data. These splits are used to train and evaluate the sentiment analysis model in the subsequent steps.

Step 9: TF-IDF Vectorization

In order to convert the textual data into a numerical representation suitable for machine learning models, the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique is applied. This step involves transforming the text data into a matrix of TF-IDF features, where each row represents a document (movie review) and each column represents a unique word in the corpus.

To perform TF-IDF vectorization, the scikit-learn library provides the TfidfVectorizer() class. This class initializes a TF-IDF vectorizer object, which is capable of calculating the TF-IDF weights for each word in the corpus.

The TF-IDF vectorizer is fitted on the training data using the fit() method. This step allows the vectorizer to learn the vocabulary and IDF (Inverse Document Frequency) values from the training set. The training data is passed as input to the fit() method.

After fitting the vectorizer, both the training and testing data are transformed into TF-IDF vector representations using the transform() method. This step applies the learned vocabulary and IDF values to convert the raw text data into TF-IDF features. The transformed data can then be used as input for training the sentiment analysis model and making predictions.

In the above code, the TfidfVectorizer() class is imported from sklearn.feature_extraction.text. An instance of the TF-IDF vectorizer is created and assigned to tfidf_vectorizer. The vectorizer is then fitted on the training data by calling the fit() method with X_train as the input.

After fitting the vectorizer, the transform() method is used to convert both the training data (X_train) and testing data (X_test) into their respective TF-IDF vector representations. The transformed data is stored in X_train_vectorized and X_test_vectorized, which will be used as input for training the sentiment analysis model and making predictions, respectively.

By performing TF-IDF vectorization, the textual data is transformed into numerical features that capture the importance of words within each movie review, enabling the machine learning model to learn patterns and make predictions based on these features.

Step 10: Logistic Regression Model Training

To perform sentiment analysis on the movie reviews, a logistic regression model is employed as the classification algorithm. Logistic regression is a popular choice for binary classification tasks, where the objective is to predict one of two possible classes: positive or negative sentiment.

In this step, a logistic regression model is initialized using the LogisticRegression() class from scikit-learn. The model is then trained on the training data using the fit() method, with X_train_vectorized as the vectorized training data and y_train as the corresponding labels.

The logistic regression model learns to find the optimal decision boundary that separates positive and negative sentiments based on the TF-IDF features extracted from the movie reviews. During training, the model adjusts its internal parameters to minimize the classification error and maximize the prediction accuracy on the training set.

In the above code, the LogisticRegression() class is imported from sklearn.linear_model. An instance of the logistic regression model is created and assigned to model. The model is then trained by calling the fit() method, with X_train_vectorized as the input features and y_train as the corresponding labels.

After training, the logistic regression model learns the underlying patterns and relationships between the TF-IDF features and the sentiment labels. It can then be used to make predictions on new, unseen movie reviews.

The trained logistic regression model serves as a powerful tool for sentiment analysis, enabling the classification of movie reviews into positive or negative sentiments based on the learned patterns from the training data.

Step 11: Model Evaluation

After training the logistic regression model on the training data, it’s important to evaluate its performance on unseen data to assess its effectiveness in sentiment analysis. In this step, the trained model is used to predict labels for the test data, and the accuracy of the model is calculated.

To predict labels for the test data, the predict() method is applied to the vectorized test data, X_test_vectorized. This generates predicted labels for the corresponding movie reviews in the test set.

Next, the accuracy of the model is computed by comparing the predicted labels with the actual labels from the test set. The accuracy_score() function from scikit-learn is used for this purpose. It calculates the accuracy as the proportion of correctly predicted labels to the total number of samples.

In the above code, the accuracy_score() function is imported from sklearn.metrics. The predict() method is applied to the vectorized test data, X_test_vectorized, generating the predicted labels stored in y_pred. The accuracy is then computed by comparing the predicted labels y_pred with the actual labels y_test.

The calculated accuracy provides an indication of how well the logistic regression model generalizes to unseen movie reviews. It represents the percentage of correctly classified reviews among all the test samples. Higher accuracy values indicate better performance and reliability of the model in sentiment analysis tasks.

Step 12: Saving the Trained Model

To preserve the trained logistic regression model for future use without having to retrain it, it can be saved to a file. In this step, we will save the trained model to a file named ‘sentiment_model.pkl’ using the pickle module in Python.

The pickle module provides functionality for serializing and deserializing Python objects. We can use it to save the trained model as a binary file that can be loaded later to make predictions on new data.

In the above code, the pickle module is imported, and the filename for the saved model is defined as 'sentiment_model.pkl'. The open() function is used with the 'wb' (write binary) mode to create a new file for writing the serialized model.

The pickle.dump() function is then called, where the trained model model is passed as the first argument, and the file object file is passed as the second argument. This saves the model to the specified file.

Step 13: Making Predictions on New Text

After training and evaluating the sentiment analysis model, we can utilize it to make predictions on new text data. In this step, we will demonstrate how to preprocess the input text, transform it using the TF-IDF vectorizer, and predict the sentiment label using the trained logistic regression model.

In the above code, we first preprocess the input text using the preprocess_text() function, which cleans and prepares the text by removing punctuation, converting it to lowercase, and applying stemming if necessary.

Next, we transform the preprocessed text into a vector representation using the TF-IDF vectorizer. We use the transform() method of the vectorizer and pass the preprocessed text wrapped in a list ([preprocessed_text]) to create a vectorized representation of the text.

Once we have the vectorized representation of the new text, we can make predictions using the trained logistic regression model. The predict() method of the model is called the vectorized text data (new_text_vectorized) as the input.

We print the predicted label to indicate whether it corresponds to a negative or positive sentiment. If the predicted label is 0, we print “Negative sentiment”, otherwise, we print “Positive sentiment”.

Step 14: Performance Metrics

In addition to accuracy, it is beneficial to evaluate the model’s performance using other metrics such as precision, recall, and F1-score. These metrics provide a more comprehensive understanding of the model’s effectiveness.

To calculate these metrics, we can utilize functions from the sklearn.metrics module. Specifically, the precision_score(), recall_score(), and f1_score() functions can be employed. These functions require both the true labels and the predicted labels as inputs.

## Conclusion
In this project, we performed sentiment analysis on IMDB movie reviews using machine learning techniques. The objective was to develop a model that could accurately predict the sentiment (positive or negative) expressed in the reviews.

After performing data cleaning and preparation, we utilized the CountVectorizer and TF-IDF vectorization techniques to transform the text data into numerical features. We trained a Logistic Regression model on the preprocessed data and evaluated its performance using various metrics.

The trained model demonstrated impressive results, achieving an accuracy of 89.25% on the test data. Furthermore, we calculated additional performance metrics such as precision, recall, and F1-score, which further validated the model’s effectiveness. The precision, recall, and F1-score were calculated to be 0.8858, 0.9033, and 0.8945, respectively.

These results highlight the model’s ability to accurately classify sentiment based on IMDB movie reviews. The high accuracy and balanced precision and recall scores indicate that the model is robust and capable of effectively identifying both positive and negative sentiments expressed in the reviews.

The application of sentiment analysis in the movie industry holds significant potential. Understanding audience sentiment can provide valuable insights for movie producers, distributors, and marketers. It can help in assessing audience reactions, evaluating movie success, and informing marketing strategies. By analyzing large volumes of reviews, movie industry stakeholders can gain a deeper understanding of audience preferences and sentiment trends.

While this sentiment analysis project has demonstrated promising results, there are a few limitations to consider. The model’s performance may vary depending on the dataset and the specific domain. Additionally, the accuracy of the model can be further improved by exploring advanced natural language processing techniques, incorporating more sophisticated algorithms, and fine-tuning hyperparameters.

In conclusion, this sentiment analysis project showcases the power of machine learning in analyzing IMDB movie reviews to predict sentiment. The model’s ability to accurately classify sentiment provides valuable insights for the movie industry and opens avenues for further research and development in this field.

Visit my portfolio: https://dataexplorewithyani.my.canva.site/

BI portfolio: https://www.novypro.com/profile_projects/trihandayani

LinkedIn profile: http://www.linkedin.com/in/tri-handayani007
