# Quora_Insincere_Questions_Classification

Link: https://www.kaggle.com/c/quora-insincere-questions-classification

An existential problem for any major website today is how to handle toxic and divisive content. Quora wants to tackle this problem head-on to keep their platform a place where users can feel safe sharing their knowledge with the world

Quora is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers.

In this competition, Kagglers will develop models that identify and flag insincere questions. To date, Quora has employed both machine learning and manual review to address this problem. With your help, they can develop more scalable methods to detect toxic and misleading content.

# Word Embedding using Glove : Dictionary of word and its coefficients
GlobalVectors (GloVe) is a model that learns vectors or words from their co-occurrence information. GloVe is a count-based model. This model that learns vectors or words from their co-occurrence information, i.e. how frequently they appear together in large text corpora, is GlobalVectors (GloVe).

Count-based models learn vectors by doing dimensionality reduction on a co-occurrence counts matrix. First they construct a large matrix of co-occurrence information, which contains the information on how frequently each “word” (stored in rows), is seen in some “context” (the columns). The number of “contexts” needs be large, since it is essentially combinatorial in size. Afterwards they factorize this matrix to yield a lower-dimensional matrix of words and features, where each row yields a vector representation for each word. It is achieved by minimizing a “reconstruction loss” which looks for lower-dimensional representations that can explain the variance in the high-dimensional data.

In the case of GloVe, the counts matrix is preprocessed by normalizing the counts and log-smoothing them. Compared to word2vec, GloVe allows for parallel implementation, which means that it’s easier to train over more data. It is believed (GloVe) to combine the benefits of the word2vec skip-gram model in the word analogy tasks, with those of matrix factorization methods exploiting global statistical information.

Reference:

https://www.kdnuggets.com/2018/08/word-vectors-nlp-glove.html

https://nlp.stanford.edu/projects/glove/

# Build LSTM model to predict insincer questions classification using keras

# RNN Architecture:

1) Sequential() : Initialize RNN.

2) Add 4 layers, with 100 units in each layer

3) units : no of memory units you want to have in LSTM or number of LSTM cells

4) return_sequences will be set to "True" because we are building stacked RNN with multiple layers. If you want to add new LSTM layer after current layer then return_sequences = True and if it is last layer then return_sequences will be set to False

5) input_shape : Shape of x_train, but here we need not to give 3D shape, only shape corresponding to timestamps(2nd) and indicators(3rd) are needed. Shape corresponding to observation(1st) will automatically taken into account.

![1](https://user-images.githubusercontent.com/30834801/50140682-969d0400-02cb-11e9-8a15-5f4eb7bb1c29.PNG)

# Training Model
![2](https://user-images.githubusercontent.com/30834801/50140733-be8c6780-02cb-11e9-97f1-2b5715155b80.PNG)


# Save model:
In case you dont want to train model again you use saved model.To do so download model.json and model.h5 is same folder, recompile and run it.
After compilation is will be ready for prediction.

Keras provides the ability to describe any model using JSON format with a to_json() function. This can be saved to file and later loaded via the model_from_json() function that will create a new model from the JSON specification.
The weights are saved directly from the model using the save_weights() function and later loaded using the symmetrical load_weights() function.
The model is then converted to JSON format and written to model.json in the local directory. The network weights are written to model.h5 in the local directory

# Prediction
Prediction of model is present in Output.csv file
