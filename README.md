Unit 21 Homework: Charity Funding Predictor

## Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

* **EIN** and **NAME**—Identification columns
* **APPLICATION_TYPE**—Alphabet Soup application type
* **AFFILIATION**—Affiliated sector of industry
* **CLASSIFICATION**—Government organization classification
* **USE_CASE**—Use case for funding
* **ORGANIZATION**—Organization type
* **STATUS**—Active status
* **INCOME_AMT**—Income classification
* **SPECIAL_CONSIDERATIONS**—Special consideration for application
* **ASK_AMT**—Funding amount requested
* **IS_SUCCESSFUL**—Was the money used effectively

## Objective: Predict if the applicant will be successful if funded by Alphabet Soup.

### Step 1: Preprocess the Data

1. Read in the charity_data.csv to a Pandas DataFrame.
  * **IS_SUCCESSFUL** is the target for the model
  * The other columns are the features being used to predict if the applicant will be successful if funded by Alphabet Soup.
2. Drop the `EIN` and `NAME` columns.
3. Determine the number of unique values for each column.
4. For columns that have more than 10 unique values, determine the number of data points for each unique value.
5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.
  *Binned rare categorical variables together as "other"
  * APPLICATION_TYPE value cutoff point was 15, such that 'T17', 'T15','T29','T14', and 'T25' were binned as "other"
  * CLASSIFICATION value cutoff point was 9, such that all classifications with nine or less values were binned as "other"
6. Use `OneHotEncoder` to encode categorical variables.

### Step 2: Compile, Train, and Evaluate the Neural Network Model

1. Continue using the Jupyter Notebook in which you performed the preprocessing steps from Step 1.
2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
  * Chose 8 neurons in the first layer and 5 neurons in the second layer.
3. Create the first hidden layer with a relu activation function.
4. If necessary, add a second hidden layer with a relu activation function.
5. Create an output layer with a sigmoid activation function.
6. Check the structure of the model.
7. Compile and train the model.
8. Create a callback that saves the model's weights every epoch.
9. Evaluate the model using the test data to determine the loss and accuracy.

Loss: 0.5575956702232361, Accuracy: 0.7233819365501404

10. Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity.h5`.

### Step 3: Optimize the Model

Optimize your model to achieve a target predictive accuracy higher than 75%.

* Adjust the input data.
  * Drop the 'EIN' column like we did in the original model. 
  * Instead of dropping the 'NAME' column, bin the names that appear less than 5 times as "other". 
      * The charity names that repeat several times could indicate a repeat application, which may be an important feature in predicting success in their ventures
      * Remove the "SPECIAL_CONSIDERATIONS" column, since this feature does not tell me anything about my target based on the background information
  * Creating more bins for rare occurrences in columns.
    * Increased the number of values for each bin. Larger bins were created in the optimized model than the original model. 
    * APPLICATION_TYPE value cutoff was increased to <500, such that all application types with less than 500 values were binned as "other."
    * CLASSIFICATION value cutoff was increased to <100, such that all classfications with 100 or less values were binned as "other."

* The neural network itself remained the same. 

1. Create a new Jupyter Notebook file and name it `AlphabetSoupCharity_Optimzation.ipynb`.
2. Import your dependencies and read in the `charity_data.csv` to a Pandas DataFrame.
3. Preprocess the dataset like you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
5. Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity_Optimization.h5`.

### Step 4: Neural Network Model Report

## Predict if the applicant will be successful if funded by Alphabet Soup.

## Results
  * Data Preprocessing
    * The **IS_SUCCESSFUL** column (1 for yes or 0 for no) is the target for the model
    * The features of the model were initially the application type, affiliation, classification, use_case, organization, status, income_amt, special_considerations, and the ask_amt.
    * Initially, both the EIN and NAME columns were removed, because they were identification columns and not thought to be relevant features to the target. 
    * Later, in the model optimization, the NAME column was included and binned. 
    * Any NAME applicants that repeated more than five times were maintained in the dataset and all other applicants were binned as "other".  
  
* Compiling, Training, and Evaluating the Model
    * I chose 8 neurons in the first hidden layer with a relu activation function, 5 neurons in the second hidden layer with a relu activation function, and an output layer with a sigmoid activation function.
    * The initial model had the following result:
      Loss: 0.5575956702232361, Accuracy: 0.7233819365501404

    * Following optimization steps as detailed above, I was able to achieve the target model performance with the following result:
      Loss: 0.4423042833805084, Accuracy: 0.7871720194816589

    * The two main levers I found to optimize model performance was one, keeping the name column as a feature in the model, and two, increasing the bin sizes for rare occurrences in categorical values, APPLICATION_TYPE and CLASSIFICATION.

  ## Summary
  We were able to optimize the deep learning model from an initial accuracy of ~72% to ~78%. The model can correctly predict whether the applicant is successful or not over 75% of the time. 
  Including the applicant name as a feature was key in optimizing the model. Repeat applicants certainly influence the dataset and the model's ability to predict whether successful or not and should be kept in the model. 

  A random forest model could also be used to solve this classification problem, as it can handle large datasets efficiently and since it is a collection of decision trees, it should perform well in this task. The random forest model was attempted to test this hypothesis, and gave an initial accuracy of ~77% with 64 estimators allowed. 



