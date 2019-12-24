# Sports-Analytics

Finding trends in sports data using different Machine Learning models.

This project gave me useful experience in Data Cleaning from large CSV files with NaNs, and testing
multiple different models to see which fits best.

# knn_position.py

- Attempts to model NBA positions by a players' height and weight using K Nearest Neighbors Classification.
- Features used = Height, Weight (2)
- Highest recorded test accuracy = 63.7% (K = 11)
- (K = 3) test accuracy = 53.3%
- In this learning problem, higher values of K will fit the model better to the test data, but the
  resulting graph does not appear grouped as Data_with_Lables.png does.  This is because C-F and F-G
  positions fall in multiple classification groups (ex. C-F's can be amongst Centers or Forwards).
  Therefore, lower values of K represent these "intermediary" positions better since they are more
  dispersed, but overall there are more "1-position" players in our Dataset, so more C, F, and G are
  classified correctly (this can be seen in KNN Graphs).
- KNN graphs before and after classification are plotted in KNN Graphs folder.

# kmeans_position.py

- Attempts to model NBA positions by a players' height and weight using KMeans Clustering as Classification.
- Features used = Height, Weight (2)
- By using KMeans to classify our data, it only makes sense to use 5 and 3 clusters.  5 clusters attempts to
  use KMeans to assign each position in our data to a cluster, but the overlap between intermediary positions
  FG and CF and our inherently inseperable data does not allow clustering to sucessfully classify our data.
- Using 5 clusters, we reach test accuracy = 45%
- Using 3 clusters, we reach test accuracy = 52%
- 


# parse_csv.py

- Cleans large CSV files into simple, usable data for knn_position.py modeling.
- Drops NaNs, converts strings heights to integers, etc.
- Avoids parsing through CSV files every test run.

# connect_mysql.py
 
 - Taken from Matei Lupu (mlupu98).
 - Currently unused in development.
 - Connects to MySql databases using python packages.
 - Avoids parsing through CSV files every test run.
