# Milestone 2 (Progress Report)


## Introduction

- Task Description: Binary classification to differentiate between human generated text (from WebText) and GPT-2 generated text (from 1542M model, configuration Topk-40).
- 1542M indicates the type of GPT-2 model and k40 is the configuration of this model.
- Human generated text consists of paragraphs from the WebText dataset, which is a collection of text from blogs, papers and code curated by OpenAI.
- GPT-2 generated text are the outputs from the GPT-2 model, trained on WebText.
- The scope of the project is to perform binary classification using neural network models and provide an error analysis.


## Motivation and Contributions/Originality:

- The motivation of this project is to contribute to the field of synthetic text detection by building classification models that can differentiate between human generated text and GPT-2 generated text.
- OpenAI ‘s GPT-2 text generating model was released in 2019 and it immediately dominated the headlines. The language model has the capacity to produce an entire short story when fed with a seed like “Once upon a time”. The generated story is an amazing improvement over GPT-2’s contemporaries, other text generation algorithms.
- OpenAI initially created four language models with varying hyperparameters (i.e., 117M, 345M, 762M and 1542M).  Due to concerns of misuse like fake news generation, these models were released in stages from smallest to largest over the span of 10 months.
- The ability to analyze and detect fake news will help combat the threat of “weaponizing language models”. 
- The motivation of this project is to contribute to the field of fake news detection by building a classification model that can identify synthetically generated text. 

## Data

- Our datasets are from [the openai/gpt-2-output-dataset repository](https://github.com/openai/gpt-2-output-dataset). Due to engineering limitations, we can only use a subset of the original:
    - Webtext: first 50k out of 250k documents in json format. The genre is a collection from blogs, papers and code curated by OpenAI.
    - GPT-2: first 50k out of 250k documents in json format. GPT-2 was trained on WebText so the style of its generated text will mirror those found in Webtext.
    - Previously, we tried processing a total of 200k documents but we ran into an out-of-RAM problem.
    
- Dataset processing:
    - Since the original Webtext and GTP-2 data were provided separately and without labels, we assigned “1” to Webtext documents, “0” to GTP-2 and then concatenated the two files into one Panda DataFrame.
    - The combined 100k dataset was randomly shuffled and split into train, validation, and test datasets (60k:20k:20k), and then stored as “.csv” format.
    - We tried to store the split datasets as “.tsv” first, however one member who uses Linux had trouble reading in the “.tsv” file while “.csv” was error-free. 
    
- Dataset EDA:
    - Here’s a breakdown of each dataset by label:
    - Train -  (0: 30118, 1: 29882)
    - Validation -  (0: 9876, 1: 10124)
    - Test - (0: 10006, 1: 9994)
    - The datasets are well balanced so we do not need to deal with class-imbalance.
    - Summary statistics for 100k dataset:
        - avg words = 560
        - max words = 4301
        - min words = 2
        - average sents = 22
        - max sents = 257
        - min sents = 1
        - vocab size = 14303
    - Summary statistics for 50k GPT-2:
        - avg words = 522
        - max words = 2211
        - min words = 10
        - average sents = 21
        - max sents = 162
        - min sents = 1
    - Summary statistics for 50k Webtext:
        - avg words = 597
        - max words = 4301
        - min words = 2
        - average sents = 23
        - max sents = 257
        - min sents = 1
    - We used the same procedure described above to process a 1k dataset (500 from Webtext and 500 from GPT-2). The purpose of this subset is to test our code. The split datasets from this small subset are also nicely balanced in class.
    
    
## Engineering

- We are using a GPU provided by Google Colab. 
- In case we encounter an out-of-RAM issue when processing 100k documents, we will switch to TPU provided by Google Cloud. 
- We will be training three classifiers:
    - Logistic Regression (LR):
        - Motivation - OpenAI used LR as a baseline model so we decided to do the same. 
        - Code - adapted directly from [OpenAI](https://github.com/openai/gpt-2-output-dataset/blob/master/baseline.py). We further tuned the hyperparameter “C” and determined the best C is 100.
        - Future direction - we probably will not fine-tune LR much more. However, we are considering calculating the f-score since we did so for the other models. The original code only calculated accuracy.
        
    - Convolutional Neural Network (CNN):
        - Motivation - CNNs are very fast to train and the CNN tutorial conveniently provided a framework for binary text classification that we can adapt.
        - Code - adapted directly from the [CNN tutorial](https://github.ubc.ca/MDS-CL-2019-20/COLX_585_trends_students/blob/master/labs/Lab1/cnn_text.ipynb).
        
        - CNN model with initial weights sampled from a normal distribution - 
           CNN_Text(
           (embeddings): Embedding(146142, 300)
           (convolution_layers): ModuleList(
           (0): Conv2d(1, 32, kernel_size=(2, 300), stride=(1, 1))
           (1): Conv2d(1, 32, kernel_size=(3, 300), stride=(1, 1))
           (2): Conv2d(1, 32, kernel_size=(4, 300), stride=(1, 1))
            )
           (dropout): Dropout(p=0.5, inplace=False)
           (fc): Linear(in_features=96, out_features=2, bias=True)
            )
        - Optimizer is Adam with a learning rate of 0.0001.
        - Loss function is CrossEntropyLoss.
        - Future direction - further fine-tune hyperparameters and use pre-trained word embeddings or even ELMO as text representation.
        
    - BERT:
        - Motivation - BERT is the most cutting edge language model that we learned in class and we wanted hands-on experience with BERT.
        - Code - adapted directly from the [BERT tutorial](https://github.ubc.ca/MDS-CL-2019-20/COLX_585_trends_students/blob/master/labs/Lab2/bert_pytorch.ipynb).
        - Model - BERT-base-uncased, which is a variant of BERT with 12-layer, 768-hidden, 12-heads and 110M-parameters. It is pre-trained on lower-cased English text only. This “BERT-based-uncased” is a model identifier in Transformers so the vocabulary will be downloaded and applied to our tokenizer.
        - The maximum sequence length for BERT is 512, so we truncated any documents that are longer than 512 (i.e., only the first 512 tokens).
        - Tried 3 context representations: 
            - Pooler_output: last layer hidden state of the first classification token, [CLS]
            - Concatenate the hidden states of all 12 layers
            - Max pooling on hidden states of the first 4 layers
        - The best performance occurred when doing a pooler output on the last layer. The validation f1 score raised up to 91% after fine-tuning.

## Challenges:

- LR - since we simply used the code directly from OpenAI, we did not run into any challenges.
- CNN - the baseline model ran seamlessly after finding an appropriate learning rate. When the learning rate was 0.1, the model was not optimizing and kept producing the same validation scores for each epoch. The real challenge lies in interpreting the model. We tried following the paper [Understanding Convolutional Neural Networks for Text Classification](https://www.aclweb.org/anthology/W18-5408/) in an attempt to understand the model. The authors even provided code at their [github](https://github.com/sayaendo/interpreting-cnn-for-text). We were unsuccessful in adapting the code and are still in the process of debugging. More specifically, we are having difficulty with the `get_activations` function.
- BERT - when trying to visualize the multi-head attention weights for the first sentence in the validation set, triggered a runtime error: `RuntimeError: CUDA error: device-side assert triggered`

## Previous works

- [Understanding Convolutional Neural Networks for Text Classification](https://www.aclweb.org/anthology/W18-5408/) (2018)
    - The authors acknowledge that there are effective ways for interpreting CNNs related to images because the data is continuous. However, text data is discrete, which poses a new challenge. They believe that filters capture different semantic classes of ngrams, while activation methods, global max-pooling capture ngram features. 
    - The authors used a 1DCNN model to perform binary classification sentiment analysis. Then, they used a series of calculations to score the importance of ngrams.
    
- [Factuality Classification Using the Pre-trained Language Representation Model BERT]( http://ceur-ws.org/Vol-2421/FACT_paper_3.pdf) (2019)
    - This paper did a FACT(Factuality Analysis and Classification Task) task on plain texts that contain verbal events. They fine-tuned a pre-trained BERT model to automatically generate a factual tag for each event. The researchers achieved encouraging results, demonstrating that fine-tuning a pre-trained BERT model on text classification task is competitive and applicable.
    - The researchers used a BERT-base-multilingual model in FACT due to the fact that they were dealing with both Spanish and English texts. They chose to use pooler output - the final hidden state corresponding to [CLS] token - as the context representation, then fed it into an output layer for classification.
    - They evaluated their model using precision, recall and F1 score for each category/tag, Macro-F1 and the global accuracy. In the paper, Macro-F1 is the main measure for this task. The best result was where the Macro-F1 was 0.489 and the global accuracy was 0.622, demonstrating a good performance of their model in this FACT.
    - The authors also addressed an issue of overfitting, and suggested implementing a random seed each time as a preventative measure.
    
- [Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment]( https://arxiv.org/abs/1907.11932) (2020)
    - The researchers acknowledged that the state-of-art models are vulnerable to adversarial examples that have nuances from the original version. If models were fed maliciously crafted adversarial examples, the robustness of these models can be greatly reduced. In this paper, the researchers applied adversarial text to two NLP tasks, text classification and textual entailment, and then evaluated the performance of three models: pre-trained BERT, CNN and RNN.
    - They used an architecture, TEXTFOOLER, to introduce adversarial texts that replaced original words with new ones that have similar meanings. TEXTFOOLER successfully attacked the models with a limited number of modifications on both tasks; no matter how long the text sequence is and how accurate the model is, it can reduce the accuracy to below 15% with less than 20% word perturbation ratio in most cases.
    
    
- [DocBERT: BERT for Document Classification](https://arxiv.org/abs/1904.08398) (2019)
    - The authors examined the performance of a pre-trained BERT model on document classification. Since modeling syntactic structure is less important for document classification than for typical BERT tasks, the authors wanted to evaluate the performance of a fine-tuned BERT model over typical document classification baselines, such as logistic regression and supported vector machines.
    - The researchers did both single-label and multi-label classification task on four datasets, and evaluated the model performance on the mean F1 scores for multi-label datasets and accuracy for single-label datasets, along with the corresponding standard deviation. The fine-tuned BERT model outperformed the two baseline models (LR and SVM), showing that BERT, a generative model, is effective in dealing with discriminative tasks.
    
## Evaluation and Visualization

- Initial results:
    - LR - `{'valid_accuracy': 85.79, 'test_accuracy': 87.01}`
    - CNN - `Epoch [17/20], Train Loss: 0.3497, Validation Loss: 0.4854, Validation Accuracy: 0.8125, Validation F1: 0.8095` 
    - BERT - `Epoch [4/5], Train Loss: 0.0475, Validation Loss: 0.3874, Validation Accuracy: 0.9146, Validation F1: 0.9142`
    
- Model interpretation strategies:
    - LR - perhaps adapt the code from this [blog](https://towardsdatascience.com/how-to-interpret-a-binary-logistic-regressor-with-scikit-learn-6d56c5783b49) to produce visualizations related the model’s predictions. 
    - CNN - follow the paper “Understanding Convolutional Neural Networks for Text Classification” to gain some insight into the CNN model.
    - BERT - use the information from multi-head attention to understand BERT.

- Document-level analysis for each model:
    - We can pull out two correctly predicted documents (one each from Webtext and GPT-2) and two incorrectly predicted documents. Based on what we learned about our models, we can try to analyze why the model made its predictions. 
    
- We do not have visualizations yet.

## Conclusion

- We hope to deliver two classification models, CNN and BERT, and provide a detailed model and error analysis that highlight any insights and challenges faced in the task of classifying human text from GPT-2 text.

























