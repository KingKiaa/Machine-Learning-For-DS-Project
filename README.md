# Machine-Learning-For-DS-Project
Final Project

Kia Aalaei

Dataset: https://www.kaggle.com/datasets/seymasa/rice-dataset-gonenjasmine

I decided to choose this project because I was scrolling through all of the possible datasets to use and this one really stood out to me. I had no idea that there were so many ways to describe and measure a rice, and those were the deciding factors as to what type of rice it was. I have always had a fascination for rice due to my persian heritage, every single food in persian culture includes some sort of rice. The classification of two different types of rice stood out to me in that way. The two different rices were jasmine and gonen. The three models built were Logistic Regression, Linear SVC, and Random Forest. For each of these models, 4 experiments were done including scale features, adding of new features, preprocessing features, and noisy indicators. Below is the actual dataset features.


![image](https://user-images.githubusercontent.com/120366695/232713374-d36d695b-d108-437d-bfc4-538000f02450.png)

![image](https://user-images.githubusercontent.com/120366695/232713506-2b5a6ca3-a280-4b6c-9a01-2956357736a9.png)

The first was to divide the dataset into features and labels, this is due to the fact that I know what classification each row goes into. I already know the data. I give features the variable x and labels the variable y. I use the drop() function to drop the columns ID and class for my features, this is because I don't want the ID numbers as a part of my testing data. I also do not want the Class column because that is my label that corresponds to the specific features.

![image](https://user-images.githubusercontent.com/120366695/232713664-14f30228-bcbc-4b53-a9a4-6961c6454da3.png)

And for the labels y, I just need the Class column.

![image](https://user-images.githubusercontent.com/120366695/232713755-ac2b6efc-56ed-45bb-822b-7e34f0d422ed.png)

After the creation of the features and labels, I want to split my dataset into training and testing, and I chose 33% of the dataset to be used for testing, 67% of the dataset will be used for training. I use the train_test_split() function to split the dataset and create different test and train variables for x and y.

![image](https://user-images.githubusercontent.com/120366695/233895253-ff722ffa-745f-4958-aeb8-b6bf71e28b4b.png)

I then intialized the Linear SVC.

![image](https://user-images.githubusercontent.com/120366695/233895312-64787611-2119-4304-8cd3-44d299951ed3.png)

The next step is to fit the model on training data and get a training score based on that. I first fit the training data x and y using the function. Then to check the score we can use two different ways of doing it, Linear SVC score function or the Cross Validation Score function. Both methods gave me a score of around 98.87%, which is incredibly good.

![image](https://user-images.githubusercontent.com/120366695/233895376-61426435-4c29-4b80-bddf-be6c1ba59fe1.png)

Next, I can predict my test data by using the model, by using the predict function. I also created a confusion matrix to compare the tested and predicted y. From the confusion matrix, Jasmine is 1 and Gonen is 0. If the true value is Gonen, the prediction had around 2,700 cases where it was right and about 48 cases where it was wrong. If the true value is Jasmine, it predicted it right around 3,200 times and wrong around 23 times. So overall, the prediction ability of the Linear SVC is quite impressive.

![image](https://user-images.githubusercontent.com/120366695/233895501-af5cd92a-3f98-4167-8fde-869058d7efbd.png)

Confusion Matrix and the classification report are the two types of baseline performance that are used to test these models.

![image](https://user-images.githubusercontent.com/120366695/233895639-0af13aee-ab2b-418f-b5f6-15327f6de64a.png)

The Logistic Regression and Random Forest models follow the same exact code to the Linear SVC, with just one distinction. Instead of initializing and importing the Linear SVC, the respective model is imported, initialized, and used. Below are two code snippets that highlight the importation and initialization of the Logistic Regression and Random Forest models.

![image](https://user-images.githubusercontent.com/120366695/233896170-d2922a39-a4bd-4cb3-b314-64563a959f18.png)

![image](https://user-images.githubusercontent.com/120366695/233896113-18e8d250-f9b7-4466-8cd6-472494b9522e.png)

Since the dataset used for this project was a basic binary classification, the distinction between jasmine and gonen rice are very clear, therefore not a big difference between accuracy scores will be present when comparing the different models and their respective features experiments. This is why at the end of this report, I will include the confusion matrices of all of the experiments done and compare/ contrast to find the best model with feature experiments.

The first feature experiment is the scaling of features. The values of the features table will be added by 2, multiplied by 2, raised to the power of 2, added by 3, multiplied by 3, raised to the power of 3, added by 4, multiplied by 4, raised to the power of 4, and added by 5. There are 10 features and every column gets scaled respectively. Below is the code used to scale the features of the models, this was done for all three models, retrained and re-evaluated using the confusion matrix. All of the code for the models will be posted in this GitHub.

![image](https://user-images.githubusercontent.com/120366695/233897354-b157cdd4-a608-47e6-accc-770c3474adb1.png)

Next came the addition of new features, we are generating new features on the models by adding polynomial features (2nd and 3rd order). Below is the code used by all models to apply this feature. the new updated dataset was placed in Z, and from here on instead of X, the code will use Z. The confusion matrix will be shown at the end of this report.

![image](https://user-images.githubusercontent.com/120366695/233897602-9ff9eda4-1bba-4dab-bada-f8a5bbcf8459.png)

Next came Proprocessing Features. The scaling of the features occurs and the features get replaced with StandardScaler(). First it was imported from sklearn.preprocessing, then fit only on the train data and used from there on out. Below is the code that was used for all three methods to use StandardScaler(). Everthing was retrained and re-evaluated and the confusion matrix created.

![image](https://user-images.githubusercontent.com/120366695/233897992-65750be6-9b4f-4305-a7ef-ae1083c71de0.png)

The final feature experiment was the Noisy Indicators, where 2 random features or columns were added to the dataset: a continuous one and a discrete one. The first column created was the discrete, I chose an arbitrary value 120 to put into the dataset. For the continuous, I chose random numbers from 100 to 200 and placed them into the dataset. This was done for all three models. Everthing was retrained and re-evaluated and the confusion matrix created.

![image](https://user-images.githubusercontent.com/120366695/233898338-70f21056-b563-4dcc-ba0d-c66deb5e5a3e.png)

All of these feature experiments did well, this is due to the simplicity of the binary classification. The confusion matrices are now shown below comparing all of the models from their base version to all of the feature experimentation.

![image](https://user-images.githubusercontent.com/120366695/233899259-a303de4b-5dd2-4c49-8da0-ed1cc059400d.png)

![image](https://user-images.githubusercontent.com/120366695/233899322-61c503c8-7e47-43d5-9f61-692ff3cce2cb.png)

![image](https://user-images.githubusercontent.com/120366695/233899346-c3cb7c5d-b6bd-413a-8217-95484d7ebb76.png)

From left to right, this is the order of the feature experimentations that match these confusion matrices: Base, 1. Scale Features, 2. Addition of Features, 3. Preprocessing Features, and 4. Noisy Indicators.

I will first compare the matrices model by model, and see which feature experiment yielded the best model, and then I will compare those with the other models, to find an overall best model to use and recommend that one.
