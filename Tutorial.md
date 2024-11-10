## How To Use This Website
Hopefully our code has been written correctly and it shouldnt raise too many questions.
Follow these steps to use the website and model effectively:
1. **Load CSV File**
   GO to the **Datasets Page** and upload a csv file  (Note that we used the standard csv format which uses "," as the delimiter)

2. **Select Dataset**
    Select the dataset of choice on the **Modelling Page**
3. **Choose Input and Target Features**
    Select your input features, and the target features
4. **Model Type Selection**
   Based on the target value, you can choose a model:
    - Regression for numerical data
    - Classification for categorical data
5. **Configure your Model Parameters**
    You can adjust model parameters such as the **split ratio** (How you split the data into test and train data) and the **number of neighbors** (k, used for KNN models)
6. **Run the Model**
    Running the model will display all Metrics, resulting from the data and model
7. **ADDITIONAL FEATURE**
   As an additional feature, you can now choose your own input data to check the predicted target value
   without the need for an entire file later. Note, this only works on continuous input features

8. **Save the Pipeline**
   You can save the model, give your own name and version to it and press **Save Pipeline**

9. **Deploy Model**
    - Go to the **Deployment page**, you can select the pipeline you saved
    - Upload a new csv file (Make sure the format matches the data on which the model has been trained)
    - You can see the newly predicted when you press **Execute Pipeline on this data**