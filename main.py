#importing some Python libraries
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

features = ['Year', 'Month', 'Day', 'Hour', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)',
            'Visibility (km)', 'Pressure (millibars)', 'Temperature (C)']

#reading the training data
#reading the test data
#hint: to convert values to float while reading them -> np.array(df_test.values)[:,-1].astype('f')
df_test = pd.read_csv('weather_test.csv')
df_test['Formatted Date'] = pd.to_datetime(pd.read_csv('weather_test.csv')['Formatted Date'], format='%Y-%m-%d %H:%M:%S.%f %z')
df_test['Year'] = df_test['Formatted Date'].apply(lambda x: x.year)
df_test['Month'] = df_test['Formatted Date'].apply(lambda x: x.month)
df_test['Day'] = df_test['Formatted Date'].apply(lambda x: x.day)
df_test['Hour'] = df_test['Formatted Date'].apply(lambda x: x.hour)
df_test = df_test.astype({'Wind Bearing (degrees)': 'float', 'Year': 'float', 'Month': 'float', 'Day': 'float', 'Hour': 'float'})
df_test = df_test.drop('Formatted Date', axis=1)

df_train = pd.read_csv('weather_training.csv')
df_train['Formatted Date'] = pd.to_datetime(pd.read_csv('weather_training.csv')['Formatted Date'], format='%Y-%m-%d %H:%M:%S.%f %z')
df_train['Year'] = df_train['Formatted Date'].apply(lambda x: x.year)
df_train['Month'] = df_train['Formatted Date'].apply(lambda x: x.month)
df_train['Day'] = df_train['Formatted Date'].apply(lambda x: x.day)
df_train['Hour'] = df_train['Formatted Date'].apply(lambda x: x.hour)
df_train = df_train.astype({'Wind Bearing (degrees)': 'float', 'Year': 'float', 'Month': 'float', 'Day': 'float', 'Hour': 'float'})
df_train = df_train.drop('Formatted Date', axis=1)

max_accuracy = 0
#loop over the hyperparameter values (k, p, and w) ok KNN
for k in k_values:
    for p in p_values:
        for w in w_values:
            #fitting the knn to the data
            X_training = df_train[features[:len(features)-1]]
            y_training = df_train[features[len(features)-1]]

            X_test = df_test[features[:len(features)-1]]
            y_test = df_test[features[len(features)-1]]

            #fitting the knn to the data
            clf = KNeighborsRegressor(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(X_training, y_training)

            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            y_prediction = clf.predict(X_test)
            accurate = 0
            for i in range(len(y_prediction)):
                difference = 100 * (abs(y_prediction[i] - y_test[i]) / y_test[i])
                if 15 > difference > -15:
                    accurate += 1

            accuracy = accurate / len(y_prediction)

            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            max_accuracy = max(max_accuracy, accuracy)
            print(f'Highest KNN accuracy so far: {max_accuracy}, Parameters: k={k}, p={p}, w={w}')