# Ecoacoustic-Embeddings-using-Deep-Learning

This model is included in a research paper currently accepted for publication at The Journal of Acoustic Society of America.
<br>
<br>
[Temporal patterns in Malaysian rainforest soundscapes demonstrated using acoustic indices and deep embeddings trained on time-of-day estimation](https://pubs.aip.org/asa/jasa/article/157/1/1/3329293/Temporal-patterns-in-Malaysian-rainforest)

This repository provides the training and evaluation code for a deep learning model on eco-acoustics. The regression model uses CNN Architecture to predict time of recording of soundscapes. It was developed to take audio recordings of eco-systems to learn ecological features. The Main_Training_Code was used to train the model on the dataset and the Evaluation_Code was used to generate master spreadsheets of the activation values of each neuron in the Adaptive Max Pool Layer.

![Model Architecture](https://github.com/SamienShaheed/Ecoacoustic-Embeddings-using-Deep-Learning/blob/main/Figures/model_architecture.jpg)

The output being predicted are sine and cosine values of the predicted hours to represent the cyclical nature of time. 
