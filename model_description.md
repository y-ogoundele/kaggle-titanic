# kaggle-titanic
Python script used for data exploration and prediction of survival of passengers
 
The aim of this problem is to classify passengers in two categories (survived or did not survive).
We seem to be given a somewhat balanced training set, since 38% of the passengers in the training set survived. I tried solving it with a baseline model using Logistic Regression.

Transforming features

I tried to get as much information as possible out of the given training set. In most columns, the data was already clean, except for the “Age” and the port of embarkation columns which had data missing and the cabin number column which had data missing as well as a non standard format for all entries.

Roughly 20% of all passenger entries had null values in the “Age” column. Since this is a sizeable part of all entries, it is best to keep them, instead of simply dropping them and losing relevant data. One basic approach I came up with was to replace the missing data with the average age of passengers of the Titanic. I then decided to use a more refined workaround, by using the title of passengers. I extracted the title of each passenger from the “Name” column, using a function in the preprocess.py script, then replaced the age of passengers for whom it was missing by the average age of passengers with the same title. This technique was used for the four most frequent titles onboard the Titanic (Mr, Mrs, Miss and Master). For the passengers whose titles were not that frequent, I took the average age of passengers of the same gender.

To deal with passengers’ cabin numbers, I tried to get hold of the deck letter (‘Block’ feature) of passengers who had a cabin. Some passengers had several cabins, but in all cases they were on the same deck. Roughly 75% of passengers in the training set had a null value set as cabin number, which is coherent with the fact that less than 25% of passengers where staying in 1st class. Their cabin letter was set as “U” (for unknown).

The port of embarkation did not come across as a relevant predictor for survival, so was later dropped from the training set.

Exploratory analysis

The conclusions I drew from the exploratory analysis were that:
-The class of the passenger aboard the ship is a good predictor of survival
-The gender of the passenger is a good predictor of survival
-The number of siblings/ spouses and the number of parents/children are good predictors of survival
-The deck letter is a good predictor of survival. One hypothesis might be that people closest to the lifeboats had a better chance of survival
-The age of the passenger is not a great predictor of survival. Instead, I used age buckets ranging from “infant” to “elderly”, which were better predictors of survival.
-I did not use the fare the passenger paid to board the ship because I considered it to be a proxy of the passenger’s class

Feature selection

Based on the aforementioned exploratory analysis, I decide to pick the following features to perform the prediction: # of siblings/spouse onboard (SibSp), # of parents/children onboard (Parch), gender (Sex), deck letter (Block), age bucket (Age) and passenger class (Pclass)
Some features (gender, deck letter, age bucket and passenger class) had to be encoded into binary variables with the get_dummies pandas function.

Model evaluation

The classifier I picked was Logistic regression. To first evaluate the accuracy of the model, I split the training dataset in two, using 30% of the training set to create a test set. The accuracy I got was 0.81. Then, I used k-folds cross-validation (with k=10) to make sure the model generalised well. I reached an average of accuracy of 0.80 +/- 0,06. Finally, I ran the model on the test dataset provided by Kaggle, and got a 0.75 accuracy score.

75% accuracy is more than satisfying for a baseline model. One way to increase accuracy might be to include fare data (assumed to be a proxy of the “Pclass” feature) or to redefine the age buckets. Another way might be to use a more complex model, for example a tree based model.
