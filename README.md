__RAG-based Question Answering System: Project Overview__

This project is a Question Answering (QA) system built using the Retrieval-Augmented Generation (RAG) framework with the Llama2 model. It allows users to ask questions about data stored in a CSV file and receive relevant answers.

Key Project Components:

* __*rag_utils.py:*__  This file contains the core utility functions for the project.
    * Data Loading: load_data(file_path) reads the data from a specified CSV file.
    * Model Loading: load_model(model_name) and load_rag_model(model_name) load the Llama2 and RAG models along with their respective tokenizers.
    * Data Vectorization: vectorize_data(data, tokenizer) transforms the CSV data into embeddings for efficient retrieval.
    * Answer Generation: answer_question(embeddings, question, rag_model, rag_tokenizer) uses the RAG model to generate answers to user questions.
    * Question Prompting: myQuestionIs() prompts the user for a question.

* __*main.py:*__  This is the main script that orchestrates the entire QA process.
    * Imports: Imports the necessary functions from rag_utils.py.
    * Data Loading: Calls the load_data function to load the CSV data.
    * Model Loading: Calls the load_model and load_rag_model functions to load the necessary models.
    * Data Vectorization: Calls the vectorize_data function to create embeddings for the data.
    * Question-Answer Loop:
        * Continuously prompts the user for questions using myQuestionIs.
        * Retrieves answers using answer_question.
        * Prints the answers.
        * Exits if the user types "exit".

__How to Load and Combine Additional CSV Data:__

1. Modify load_data Function:
    * Open the `rag_utils.py` file.
    * Edit the `load_data` function to:
        * Take an additional argument for the second file path.
        * Load both CSV files into separate DataFrames.
        * Concatenate the DataFrames using pd.concat.
        * Return the combined DataFrame.
2. Update main.py Script:
    * In the `main.py` file, provide the path to the second CSV file when calling the load_data function.
    * The rest of the code should work seamlessly with the combined data.

Example Code Changes in rag_utils.py:

``` 
def load_data(file_path1, file_path2):
    Loads two CSV files into a pandas dataframe
    data1 = pd.read_csv(file_path1)
    data2 = pd.read_csv(file_path2)
    combined_data = pd.concat([data1, data2])
    return combined_data
``` 

Example Code Changes in `main.py` :
``` 
# ... other imports ...
file_path1 = 'data.csv'
file_path2 = 'additional_data.csv'
data = load_data(file_path1, file_path2)
# ... rest of the main function ...

``` 

Important Considerations:

* Ensure that the columns in both CSV files are compatible for concatenation.
* You might need to adjust data preprocessing steps if the combined data requires it
