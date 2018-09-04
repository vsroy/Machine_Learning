#Program to implement Gradient Descent Algorithm
from numpy import *
#m = slope, b = y-intercept
def Compute_Error_For_Line_Given_Points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m*x + b))*2
    return (totalError/float(len(points)))

def Step_Gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N=float(len(points))

    for i in range(0, len(points)):
        x=points[0, 1]
        y=points[i, 1]
        b_gradient += -2/N *(y-((m_current*x) + b_current))
        m_gradient += -2/N*x*(y-((m_current*x) + b_current))
        new_b = b_current - learning_rate*b_gradient
        new_m = m_current - learning_rate*m_current
    return([new_b, new_m])

def GradientDescentRunner(points, starting_b, starting_m, learning_rate, iterations):
    b = starting_b
    m = starting_m
    for i in range(iterations):
        b, m = Step_Gradient(b, m, array(points), learning_rate)
    return(b, m)

def Run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 #guessting initial x intercept
    initial_m = 0 #guessing initial slope
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m,
                                                                              Compute_Error_For_Line_Given_Points(initial_b,
                                                                                                            initial_m,
                                                                                                            points)))
    print("Running.....")
    [b, m] = GradientDescentRunner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m,
                                                                Compute_Error_For_Line_Given_Points(b, m, points)))

if __name__ == '__main__':
    Run()




