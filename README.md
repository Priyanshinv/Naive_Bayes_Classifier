# Naive_Bayes_Classifier
Classifier for whether a given chunk of text describes a commercial establishment that offers office space for rent, in python 
Used Naive Bayes Classifier for classification (based on probabilities)
Steps involved are:
•	Data Cleaning: 
Removed special characters and digits using the regex: [^a-zA-Z ]+ This removed all the characters that are not alphabets
•	Splitting the Data Set: 
Split the dataset (total of 100 entries) into training set and testing set in the ratio 9:1 
•	Probabilities Calculation:
For every chunk in testing set, we need to find the following probabilities:
Probability of not offering office space for rent, given the text:
P(0|text) = (P(text|0) *  P(0))/P(text)
Probability of offering office space for rent, given the text:
P(1|text) = (P(text|1) * P(1))/P(text)

We can get the P(0) and P(1) from training data set as follows:
P(0) = Number of texts in 0 class/ Total number of texts in the set
P(1) = Number of texts in 1 class/ Total number of texts in the set

P(text|0) = P(word1|0) * P(word2|0) * … P(wordn|0) where word1, word2, … wordn constitute text
P(text|1) = P(word1|1) * P(word2|1) * … P(wordn|1) where word1, word2, … wordn constitute text

P(word|class) = Number of times the word appears in the class/ Total number of words in the class

To get the total number of words in each class we can use CountVectorizer from sklearn which returns a term-document matrix (TDM) consisting of a list of word frequencies appearing in a set of documents. We can then calculate probabilities of each word for each class. However, a few words can have probability of 0, which would make the entire calculations equal to 0. To avoid this, we use Laplace Smoothing (line no 84 and 85 of source code)
•	Assignment of class
Finally, for each text we check the greater of the two – P(0|text) and P(1|text) and if P(0|text) > P(1|text) we assign its class to be  0 i.e., the chunk of text describes a commercial that doesn’t offer office space as rent, else we assign its class to be 1 i.e., the chunk of text describes a commercial that offers office space as rent.

Result:
Accuracy of classifier is 70%
