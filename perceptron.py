import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

W = [1,-0.5,1]

y = lambda x: (-(W[0]/W[1]))*x + (-W[2]/W[1])

# PLOTS LINEAR BOUNDARY
def plot_line(x_data_points):
    x_values = [i for i in range( int(min(x_data_points))-1,
                       int(max(x_data_points))+2)]

    y_values = [y(x) for x in x_values]

    plt.plot(x_values,y_values)

# PLOTS SCATTER GRAPH
def plot(X):
    for sample in X:
        point = sample[0]
        target = sample[1]

        if target == 0:
            plt.scatter(point[0], point[1],
            marker = "o", c = "r")
        else:
            plt.scatter(point[0], point[1],
            marker = "+", c = "r")


# READ DATASET FROM THE CSV FILE
def read_csv(dataset):
    data = pd.read_csv("../datasets/" + dataset)
    #print(data.head())
    #print()

    points = data.loc[ : ,['x','y']]
    points['b'] = 1
    #print(points.head())
    #print()
    targets = data.loc[ : ,['t']]
    #print(targets.head())
    #print()

    # CONVERT DATAFRAMES TO PYTHON LIST
    points_lst = points.values.tolist()
    targets_lst = targets.values.tolist()
    targets_lst = [t[0] for t in targets_lst]
    #print(points_lst[:5])
    #print(targets_lst[:5])
    #print()

    #print("Lenght Points_lst: {}".format(len(points_lst)))
    #print("Lenght targets_lst: {}".format(len(targets_lst)))
    #print()

    X = list(zip(points_lst, targets_lst))
    #print(X[:5])
    return X


# CALCULATES PREDICTION
def cal_pred(point):
    #calculate the value of prediction

    #print(W[0]*point[0] + W[1]*point[1] + W[2]*point[2])
    pred = (W[0]*point[0] + W[1]*point[1] + W[2]*point[2] >= 0)
    return int(pred)


def main(argv):
    learning_rate = 0.1
    loop_again = 0
    epoch = 1

    # READ DATASET NAME FROM COMMAND LINE
    if len(argv) != 2:
        print("Usage: python perceptron.py dataset.csv")
        sys.exit(1)
    else:
        # READ DATAFROM CSV FILE
        X = read_csv(argv[1])

    while(True):
        print("Epoch: {}".format(epoch))

        print("W[0]:{}".format(W[0]))
        print("W[1]:{}".format(W[1]))
        print("W[2]:{}".format(W[2]))

        plot(X)
        plot_line([sample[0][0] for sample in X])
        plt.show()

        for sample in X:
            point = sample[0]
            target = sample[1]

            #print("Point:({},{})".format(point[0],point[1]), end="  ")

            pred = cal_pred(point)

            #print("target: {}".format(target), end="  --  ")
            #print("prediction: {}".format(pred))

            if target - pred != 0: #TRUE WHEN POINT IS INCORRECTLY CLASSIFIED
                loop_again = 1
                W[0] = W[0] + (learning_rate * (target - pred) * point[0])
                W[1] = W[1] + (learning_rate * (target - pred) * point[1])
                W[2] = W[2] + (learning_rate * (target - pred))

        if loop_again == 0:
                break

        epoch += 1
        loop_again = 0
        print("---------------------------------------------------------------")
        print()

    print()
    print("Final Weights:")
    print("W[0]:{}".format(W[0]))
    print("W[1]:{}".format(W[1]))
    print("W[2]:{}".format(W[2]))


if __name__ == '__main__':
    main(sys.argv)
