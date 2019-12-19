
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#goal of this is to map height and weight to try and predict position using height and weight of the player
#using KNN classification


#Thoughts:
#   - must use a stratified K fold since the "in between" positions seem underrepresented
#     in the training data
#   - then use cross validation to decide what k is best for all categories
#   - legend should have better colors


def encode_position(positions):
    #Create a dictionary of unique position:label pairs
    i = 0
    pos_dict = {}
    positions.sort()
    for pos in positions:
        pos_dict[pos] = i
        i = i + 1
    return pos_dict


def main():
    #remember to run parse_csv.py before trying to classify

    #import newdata, the parsed csv
    newdata = pd.read_csv('../nba-players-stats/newdata.csv')

    #encode the unique positions from newdata into usable labels
    pos_dict = encode_position(newdata.position.unique())

    #now you are left with clean data with no Nan's and extra testing values
    #plot the data on height vs weight with matplotlib
    print("Plotting Data...")
    _, ax = plt.subplots(1)
    ax.set_ylabel('Height (in)')
    ax.set_xlabel('Weight (lbs)')
    ax.set_title('Player Data')
    scatter = ax.scatter(newdata['weight'], newdata['height'], c=newdata['int_position'])
    legend = ax.legend(*scatter.legend_elements(), loc='lower right', title='Positions')
    i = 0
    for pos in pos_dict:
        legend.get_texts()[i].set_text(pos)
        i = i + 1
    ax.add_artist(legend)
    plt.savefig('../KNN Graphs/Data_with_labels.png')

    #define a classifier;  here we want to use KNN
    #k = 11 neighbors maximizes test accuracy -- highest accuracy = 63.696 %
    #smaller values of k more accurately predict C-F and F-G since the number of
    #required neighbors is less, but a higher value of k more accurately predicts
    #the dataset as a whole.  At this point, I decide to leave k large in order to
    #separate the data as cleanly as possible
    k = 11
    print("KNN Classifying with %d nearest neighbors..." % k)
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    
    #create labels and features
    targets = newdata['int_position']
    feature_vec = []
    for ___, player in newdata.iterrows():
        feature = (player['height'], player['weight'])
        feature_vec.append(feature)

    #split the data into testing and training data
    #X = features, y = labels
    X_train, X_test, y_train, y_test = train_test_split(feature_vec, targets, test_size=0.2)

    #fit the classifier to the training data
    knn_clf.fit(X_train, y_train)

    #predict training data and report accuracy
    y_train_pred = knn_clf.predict(X_train)
    train_acc = 100.0*metrics.accuracy_score(y_train_pred, y_train)
    print("\t--Training Accuracy: %.3f" % train_acc)

    #replot the training data with color == assigned label
    #print("Plotting Training Data after KNN...")
    heights = []
    weights = []
    for x in X_train:
        heights.append(x[0])
        weights.append(x[1])

    _, ax2 = plt.subplots(1)
    ax2.set_ylabel('Height (in)')
    ax2.set_xlabel('Weight (lbs)')
    ax2.set_title('Training Data After KNN')
    scatter1 = ax2.scatter(weights, heights, c=y_train_pred)
    legend1 = ax2.legend(*scatter1.legend_elements(), loc='lower right', title='Positions')
    i = 0
    for pos in pos_dict:
        legend1.get_texts()[i].set_text(pos)
        i = i + 1
    ax2.add_artist(legend1)
    plt.savefig('../KNN Graphs/Train_Data_After_KNN.png')

    #predict test values with the fitted model and report accuracy
    y_test_pred = knn_clf.predict(X_test)
    test_acc = 100.0*metrics.accuracy_score(y_test_pred, y_test)
    print("\t--Testing Accuracy: %.3f" % test_acc)
    print("Done.")

if __name__ == '__main__':
    main()