# Quora_Insincere_Questions_Classification

Link: https://www.kaggle.com/c/quora-insincere-questions-classification

An existential problem for any major website today is how to handle toxic and divisive content. Quora wants to tackle this problem head-on to keep their platform a place where users can feel safe sharing their knowledge with the world

Quora is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers.

In this competition, Kagglers will develop models that identify and flag insincere questions. To date, Quora has employed both machine learning and manual review to address this problem. With your help, they can develop more scalable methods to detect toxic and misleading content.

# Build LSTM model to predict insincer questions classification using keras

# RNN Architecture:

1) Sequential() : Initialize RNN.

2) Add 4 layers, with 100 units in each layer

3) units : no of memory units you want to have in LSTM or number of LSTM cells

4) return_sequences will be set to "True" because we are building stacked RNN with multiple layers. If you want to add new LSTM layer after current layer then return_sequences = True and if it is last layer then return_sequences will be set to False

5) input_shape : Shape of x_train, but here we need not to give 3D shape, only shape corresponding to timestamps(2nd) and indicators(3rd) are needed. Shape corresponding to observation(1st) will automatically taken into account.

# Save model:
In case you dont want to train model again you use saved model.To do so download model.json and model.h5 is same folder, recompile and run it.
After compilation is will be ready for prediction.

Keras provides the ability to describe any model using JSON format with a to_json() function. This can be saved to file and later loaded via the model_from_json() function that will create a new model from the JSON specification.
The weights are saved directly from the model using the save_weights() function and later loaded using the symmetrical load_weights() function.
The model is then converted to JSON format and written to model.json in the local directory. The network weights are written to model.h5 in the local directory
