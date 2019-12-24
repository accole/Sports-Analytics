
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, model_selection, neighbors, cluster
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

#goal of this is to map height and weight to try and predict position using height and weight of the player
#using Clustering (KMeans)

def tuplify(centroid):
    #turns an np.array centroid coordinates into a Tuple
    #We want to do this so we can hash centroids in a dictionary
    tup = (centroid[0], centroid[1])
    return tup

def encode_position(positions):
    #Create a dictionary of unique position:label pairs
    i = 0
    pos_dict = {}
    positions.sort()
    for pos in positions:
        pos_dict[pos] = i
        i = i + 1
    return pos_dict


def map_labels(groups):
    #get centroids and assigned labels from groups
    centroids = groups.cluster_centers_

    #takes randomly assigned cluster labels and maps them to our position labels
    #order the centroids to get the correct position order
    # key = centroid : value = position label
    centroid_pos_dict = {}
    #we know the order of the centroids ascends by X and Y
    #I choose to use X
    threshold_x = 350
    max_centroid = centroids[0]
    for pos in range(len(centroids)):
        for c in centroids:
            #keep centroid if it is the smallest centroid above x threshold
            if (c[1] < threshold_x) and (c[1] > max_centroid[1]):
                max_centroid = c
        
        #map the order into dictionary
        tup = tuplify(max_centroid)
        centroid_pos_dict[tup] = pos
        
        #update conditionals
        threshold_x = max_centroid[1]
        for c in centroids:
            #start next iteration at an unused centroid
            if tuplify(c) not in centroid_pos_dict.keys():
                max_centroid = c

    #if we are using 3 clusters, we must map the 3 centroids to G, F, and C
    # not FG or CF
    # 0 -> 0    1 -> 2    2 -> 4
    if (len(centroids) == 3):
        for c in centroids:
            tup = tuplify(c)
            key = centroid_pos_dict[tup]
            if (key == 1):
                centroid_pos_dict[tup] = 2
            elif (key == 2):
                centroid_pos_dict[tup] = 4

    #find which random label assigns to which centroid and map assigned label to position label
    #we can find the assigned label from the centroid by predicting the label of the centroid
    # key = centroid : value = assigned label
    centroid_assigned_dict = {}
    for c in centroids:
        tup = tuplify(c)
        predict = groups.predict(c.reshape(1,-1))
        centroid_assigned_dict[tup] = predict[0]

    #now we have centroid -> position and centroid -> assigned
    #combine these to map assigned -> position
    # key = assigned label : value = position label
    label_dict = {}
    for c in centroids:
        tup = tuplify(c)
        label_dict[centroid_assigned_dict[tup]] = centroid_pos_dict[tup]

    #return label dictionary
    return label_dict



def main():
    #open the parsed csv file
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
    plt.savefig('../Cluster Graphs/Data_with_labels.png')

    #define a classifier;  here we want to use KMeans
    #intuitively we want to have 5 centroids, 1 for each position.  Some issues I think
    #I will run into will be the centers of the clusters won't line up with the
    #natural seperation of the positions since the data is very inseperable using
    #just height and weight.
    #Using 3 clusters, however, might match the accuracy of KNN = 11 since it will
    #group the more frequent C, F, G better.  Let's take a look.
    clusters = 3
    print("KMeans Clustering with %d clusters..." % clusters)
    
    #create features and labels
    targets = newdata['int_position']
    feature_vec = []
    for ___, player in newdata.iterrows():
        feature = (player['height'], player['weight'])
        feature_vec.append(feature)

    #In Clustering, the training process is unsupervised, therefore we can cluster
    #based on our entire data set, and no overfitting will happen
    #but for the sake of testing new data points on our cluster, we split the data
    X_train, X_test, y_train, y_test = train_test_split(feature_vec, targets, test_size=0.2)

    #Cluster all the data
    groups = cluster.KMeans(n_clusters=clusters).fit(X_train)
    centroids = groups.cluster_centers_
    assigned_labels = groups.labels_

    #knowing how the KMeans algorithm runs, it randomly assigns centroids and iteratively
    #loops until the centers move very little or stop moving.  Therefore, the "assigned label"
    #that the algorithm assigns each of the data points is between 0-4, which nicely aligns
    #with our position encoding.  Since the labels randomly assign to centroids, our accuracy
    #will vary greatly based on the random cluster label assignment.  Therefore, we must map
    #between these labels before plotting / reporting accuracy.
    label_dict = map_labels(groups)

    #change the labels
    cluster_labels = []
    for label in assigned_labels:
        cluster_labels.append(label_dict[label])

    #replot the training data with color == assigned cluster
    #print("Plotting Data after KMeans...")

    heights = []
    weights = []
    for x in X_train:
        heights.append(x[0])
        weights.append(x[1])

    _, ax2 = plt.subplots(1)
    ax2.set_ylabel('Height (in)')
    ax2.set_xlabel('Weight (lbs)')
    ax2.set_title('Data after KMeans Clustering')

    #plot cluster centers as red dots and distinctly color the clusters
    scatter1 = ax2.scatter(weights, heights, c=cluster_labels)
    
    cent_height = []
    cent_weight = []
    for c in centroids:
        cent_height.append(c[0])
        cent_weight.append(c[1])

    ax2.scatter(cent_weight, cent_height, c='red')

    legend1 = ax2.legend(*scatter1.legend_elements(), loc='lower right', title='Clusters')
    if (clusters == 3):
        legend1.get_texts()[0].set_text("C")
        legend1.get_texts()[1].set_text("F")
        legend1.get_texts()[2].set_text("G")
    else:
        i = 0
        for pos in pos_dict:
            legend1.get_texts()[i].set_text(pos)
            i = i + 1
        ax2.add_artist(legend1)
    plt.savefig('../Cluster Graphs/Clustered Data.png')

    #report training accuracy of the clusters
    train_acc = 100.0*metrics.accuracy_score(cluster_labels, y_train)
    print("\t--Training Accuracy: %.3f" % train_acc)

    #Predict position of testing data using the clusters
    predicted_labels = groups.predict(X_test)

    #map the labels to position encodings
    i = 0
    for label in predicted_labels:
        predicted_labels[i] = label_dict[label]
        i = i + 1

    #report testing accuracy (if the clusters actually represent player position)
    test_acc = 100.0*metrics.accuracy_score(predicted_labels, y_test)
    print("\t--Testing Accuracy: %.3f" % test_acc)
    print("Done.")

if __name__ == '__main__':
    main()