# Text-Classification-NLTK
**OBJECTIVE:**
Classification/Prediction of texts which belongs to nine different categories/books of the Gutenberg’s digital corpus by using various ML classification algorithm and then choosing the best algorithm for the purpose depending on their performance. 

**DATA PREPARATION AND CLEANING**

The data needs to be prepared and cleaned and it is an essential part of this assignment. Nine different books are selected based on their authors from the Gutenberg's digital corpus for having nine distinct classes with the aim of having distinct targets for the model. The steps involved in the data preparation process are:

1. Retrieval and selection of texts from Gutenberg's digital corpus. The NLTK library includes a small selection of texts from the Project Gutenberg electronic text archive, which contains some 25,000 free digital books. Thus to begin with, first the NLTK library was imported. 
2. Nine distinct books/texts were selected from the Gutenberg corpus based on their authors.
3. The selected texts are then sentence tokenized and then a function was defined, named as ‘word_list’ and it has the capacity of removing unwanted punctuations and stop-words (as these would corrupt the training model and are therefore needed to be removed). The sentences were further tokenized to words and lemmatizing took place which is the process by which the all the words are reduced to the root words by using the morphological analysis of words and the vocabulary depending upon whether it’s a noun verb or an adjective. It helps in reducing the size of the data and also makes it convenient to keep track the specific words instead the same words used in different forms. Since there is a requirement for only words there’s a need to remove all data that doesn’t count as words e.g. numbers, punctuations etc. It reduces the size of the data as well as it makes the process faster. All the words are also transformed to lower case so that the computer doesn’t recognise same words in different cases as unlike.
4. As the number of words in each of these 9 texts/books are not same, another function is then defined which is responsible for breaking down a book/text into multiple documents of 100 words each, and then selecting 150 random documents for each book/text. This function is used to make sure that the trained model is not biased to any particular text because of having more words from it, thus having same number of words from each text will ensure an unbiased model at the end. This function is applied to the processed texts from the previous step and stored in variables.
5. Then we label each of this document obtained from the previous with the corresponding author's name and then we shuffle them, followed by storing them in a single dataframe. 
6. The Dataframe is then shuffled to avoid any bias and then we perform the word to vector transformation.

**WORD TO VECTOR TRANFORMATION**

Feature transformation is an approach for converting all the textual data into numeric form as the Machine Learning Algorithms work only with numeric data. Since we only have textual data available, the numeric features are extracted by using two different techniques which are Bag-of-Words (BOW) and Term Frequency-Inverse Document Frequency (TF-IDF). These are discussed below.

*Bag-of-Words (BOW)-*
The Bag-of-Words preserves the words present in the corpus. The words are referred as a feature for the documents. The frequency of each word is calculated in the current document, in this way the word features are engineered or extracted from the textual corpus. This was done by using the ‘CountVectorizer’ function, Min_df was set to 3, meaning that words which had a document frequency of less than 3 were ignored. CountVectorizer was also set to take both unigrams and bigrams into consideration. Only the top 5000 most important features based on the frequency of the words were considered.

*TF-IDF-*
The Term Frequency-Inverse Document Frequency is the most popular method to covert the textual data into numerical feature. This method helps us to highlight words which are interesting and unique in the particular document but not across all the documents. It also assigns a particular integer numbers to these each of these words.
The Term Frequency or  TF = (Frequency of the word in the sentence) / (Total number of words in the sentence)
The Inverse Document Frequency or IDF: (Total number of sentences (documents))/(Number of sentences (documents) containing the word)

**Modeling**

The Machine Learning algorithms we used and compared for this classification problem are:
1. Random Forest
2. Support Vector Machine
3. K-Nearest Neighbor
4. Decision Tree

*Cross Validation-*
The cross validation is the most important part as it’s essential to validate the stability of the machine learning model and well it would generalize the new data. The split of the data needs to be done carefully if the train data is less we risk the important patterns in the data set which in turn increases the error in the process. Hence ample data is required for the train data set, the K fold cross validation does the this exactly. 
In this process the data is divided into k subsets, and hence the same process is repeated k times. Every time one of the k subset is used as the test set the other k-1 subsets are combined to form a training data set. code the KFold is a function consists parameters for splitting the data number of times, shuffle, and the random_state
The algorithm for the cross validation is given below:
1.	Randomly split the whole data set into k fold.
2.	For each k fold in the data set, we build a model on k-1 fold of the dataset. Then it tests the model to check the effectiveness for kth fold.
3.	Check the errors for the predictions.
4.	Repeat the whole process until each of the k folds has served as the test set.
5.	The average of the k recorded errors called cross validation error and it will serve as the performance metric for the model. 

The accuracy for SVM algorithm (for both BOW and TF-IDF transformations) was the highest compared to all the other models, we have selected it as our champion model.

**Error Analysis**

Continuing with our best model (i.e. the Linear SVM), we looked at the confusion matrix in order to see the discrepancies between predicted and actual labels if any. And from the confusion matrix, we observed that most of the predictions made by the model were correct (as vast majority of the predictions were present at the diagonal). But there were a few misclassification and thus we tried to find the possible reason behind these misclassifications.


