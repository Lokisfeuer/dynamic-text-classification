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

## How to use it
At the end of both files an AI is created by the following steps.
1. Create a new dynamic AI obejct passing it a name (it will only use for nameing files)
2. Call the function .generate_training_data with multiple arguments. When using the binary version anter a true prompt and a false prompt. Then,in both versions pass a prompt_nr. This number determines how many variations of your prompt will be created. Also pass answer_nr; this number determines how many answers are given for each variation of the prompt. There are no answers given to the original prompt. In the multi class version you can also pass load=True. If you do so it will attempt to load started files with generated data from prior attempts. This is useful if openai is overloaded and gives out an error every now and then to not have to restart from the beginnning. IN the multiclass version you pass any number of prompmts as kwargs to this function.
3. You can call the function .analyse_training_data without parameters to analyse the created data.
4. Then call embed_data() which tokenizes the data and creates a dataset from it.
5. lastly call the .train function. This funtion is the only that returns somehting. It returns the training history as well as the model itself. It has multiple parameters: The amount of epochs, the learning rate, the fraction of samples of the full dataset to use as validation set, the batch size for the training batches and lastly the loss funciton that is to be used. I recommend varying mainly the epochs and the learning rate. Varying the loss function might cause errors in the multiclass version.
6. The history object returned by the train function has a .show() function which you can call the to see different metrics monitored during the training process.
