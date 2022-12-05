# importing some Python libraries
import pandas as pd
import csv
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
# import jellyfish
# print(jellyfish.levenshtein_distance('jellyfish', 'smellyfish'))


# Target Class
# Goals
# W L D


# Attributes (IP)
# Team Rating (IP)
# https://www.kaggle.com/datasets/bryanb/fifa-player-stats-database?select=FIFA17_official_data.csv
# Calculate team ratings for PL teams for seasons 17, 18, 19

# (Head to Head)
# W-col D-col L-col
# Number of times the team had each result vs. X opponent

# Last 10 games (Complete)
# points/possible points in last 10 games

# Teams = ID
# Attributes:
# = Rating
# = Form (Points in last 10)
#
# Class = ?



# defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

# Read PL Match Outcomes
pl_18_19 = pd.read_csv("season-1819_csv.csv")
pl_17_18 = pd.read_csv("season-1718_csv.csv")

fifa_ratings = pd.read_csv("FIFA18_official_data.csv")
fifa_ratings2 = pd.read_csv("FIFA17_official_data.csv")

def process_data(match_data, ratings_data):
    pass

# Sort data by most recent date
pl_18_19 = pl_18_19.iloc[::-1]
pl_17_18 = pl_17_18.iloc[::-1]

# Extract relevant columns
pl_18_19 = pl_18_19[["Date", "HomeTeam", "AwayTeam", "FTR"]]
pl_17_18 = pl_17_18[["Date", "HomeTeam", "AwayTeam", "FTR"]]

# Extract a set of the team names
teams = pl_18_19["HomeTeam"]
team_name_set = {'Arsenal'}

for name in teams:
    team_name_set.add(name)

team_names = list(team_name_set)
team_names.sort()
team_points_data = {}
team_check = {}
team_ratings = {}

for team in team_name_set:
    team_points_data[team] = 0
    team_check[team] = 0
    team_ratings[team] = []

for index in fifa_ratings.index:
    club = fifa_ratings['Club'][index]

    # closest_name = 'Name'
    # for name in team_names:
    #     distance = jellyfish.levenshtein_distance(name, str(club))
    #     if distance < 1:
    #         closest_name = name

    if club in team_names:
        # name = fifa_ratings['Name'][index]
        rating = fifa_ratings['Overall'][index]
        team_ratings[club].append(rating)

for team in team_ratings:
    team_ratings[team].sort(reverse=True)
    team_ratings[team] = team_ratings[team][:15]
    lst = team_ratings[team]
    team_overall_rating = sum(lst) / len(lst)
    team_ratings[team] = team_overall_rating

num_games = 5

for index in pl_18_19.index:
    home_team = pl_18_19['HomeTeam'][index]
    away_team = pl_18_19['AwayTeam'][index]
    ftr = pl_18_19['FTR'][index]
    date = pl_18_19['Date'][index]
    team_check[home_team] += 1
    team_check[away_team] += 1
    if 'H' in ftr:
        if team_check[home_team] <= num_games:
            team_points_data[home_team] += 3

    elif 'A' in ftr:

        if team_check[away_team] <= num_games:

            team_points_data[away_team] += 3

    else:

        if team_check[home_team] <= num_games:
            team_points_data[home_team] += 1

        if team_check[away_team] <= num_games:
            team_points_data[away_team] += 1

with open('football.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # header = ["Team", "Form"]
    # writer.writerow(header)
    # for team in team_names:
    #     print(f'{team} : {team_points_data[team]}', end=" | ")
    #     writer.writerow([team, team_points_data[team]/(num_games*3)])

    header = ["HomeForm", "HomeRating", "AwayForm", "AwayRating",  "Outcome"]
    writer.writerow(header)


    for index in pl_18_19.index:
        home_team = pl_18_19['HomeTeam'][index]
        away_team = pl_18_19['AwayTeam'][index]
        home_form = float(team_points_data[home_team]/(num_games*3))
        away_form = float(team_points_data[away_team]/(num_games*3))
        home_rating = float(team_ratings[home_team])
        away_rating = float(team_ratings[away_team])
        outcome = pl_18_19['FTR'][index]
        if 'H' in outcome:
            outcome = float(3)
        elif 'A' in outcome:
            outcome = float(2)
        elif 'D' in outcome:
            outcome = float(1)

        writer.writerow([home_form, home_rating, away_form, away_rating, outcome])
    f.close()

# def getForm(dataframe, last_n_games):
#     points = {}
#     games_read = {}
#     for index in dataframe.index:
#         home = pl_18_19['HomeTeam'][index]
#         away = pl_18_19['AwayTeam'][index]
#         ftr = pl_18_19['FTR'][index]
#         date = pl_18_19['Date'][index]
#         games_read[home] += 1
#         games_read[away] += 1
#         if 'H' in ftr:
#             if games_read[home] <= last_n_games:
#                 points[home] += 3
#
#         elif 'A' in ftr:
#
#             if games_read[away] <= last_n_games:
#                 points[away] += 3
#
#         else:
#
#             if games_read[home] <= last_n_games:
#                 points[home] += 1
#
#             if games_read[away] <= last_n_games:
#                 points[away] += 1


data = pd.read_csv("football.csv")
max_accuracy = 0
#loop over the hyperparameter values (k, p, and w) ok KNN
for k in k_values:
    for p in p_values:
        for w in w_values:
            X_cols = header[:len(header)-1]
            y_col = header[len(header)-1]
            X = data[X_cols]
            y = data[y_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=88)

            #fitting the knn to the data
            clf = KNeighborsRegressor(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(X_train, y_train)
#
            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            y_prediction = clf.predict(X_test)
            y_test = y_test.to_numpy()
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
