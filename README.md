# dynamic-text-classification
This is a project in which I tried to create a programm which would generate a neural network to classify a sentence to one topic or another.
It works like this:
1. You enter a prompt for each topic it should be able to classify.
2. This prompt is given to openais davinci-003 modul to generate data.
3. The data then gets analysed
4. and is tokenized with the BERT language modul.
5. With that the neural network can be trained and it as well as a history of the training process is being returned.
I made two different classes, one for binary classification using the BCELoss and one for multi class classification problems using the normal Cross Entropy Loss.
To generate data the program will need an API key for OpenAI.
