
import csv
import pandas as pd

#goal of this is to map height and weight to try and predict position using height and weight of the player
#using KNN classification


#Thoughts:
#   - must use a stratified K fold since the "in between" positions seem underrepresented
#     in the training data
#   - then use cross validation to decide what k is best for all categories
#   - legend should have better colors


def convert_height(str_height):
    #converts a string height rep into integer rep
    #split the string literal
    ft_in = str_height.split('-')

    #extract feet and inches in ints
    feet = int(ft_in[0])
    inch = int(ft_in[1])

    #return the height in inches
    inches = feet*12 + inch
    return inches


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
    #open the csv file and extract data using pandas
    print("Parsing Player Stats ...")
    playerdata = pd.read_csv('../nba-players-stats/player_data.csv')
    newdata = playerdata[['position', 'height', 'weight']].copy()

    #clean the data of NaNs
    #delete the rows with empty columns
    print("Cleaning Data...")
    newdata = newdata.dropna()

    #make F-C and C-F, G-F and F-G equivalent
    clean_pos = []
    for ___, player in newdata.iterrows():
        if player['position'] == "F-C":
            clean_pos.append("C-F")
        elif player['position'] == "G-F":
            clean_pos.append("F-G")
        else:
            clean_pos.append(player['position'])

    newdata['position'] = clean_pos

    #then encode the unique positions into usable labels
    pos_dict = encode_position(newdata.position.unique())

    #convert string values in data to ints
    int_height = []
    int_pos = []
    for ___, player in newdata.iterrows(): 
        #change all height literals into integers
        h = convert_height(player['height'])
        int_height.append(h)
        #encode positions to integers
        p = pos_dict[player['position']]
        int_pos.append(p)

    #replace string literal height column with integer column
    #and add encoded position column
    newdata['height'] = int_height
    newdata['int_position'] = int_pos

    print("Saving clean data to newdata.csv ...")
    newdata.to_csv("../nba-players-stats/newdata.csv")

    #done
    print("Done.")
    return

if __name__ == '__main__':
    main()