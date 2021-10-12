import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

"""
TO DO LIST: hej
Create a SVM classifier. Use transformation into higher dimension
in order to separate data and to create an indicator function ind(s).
Use dual formulation, kernel functions and slack variables.

DONE 1. Define a suitable kernel function (a function which takes two data points as arguments and returns a scalar value)'
   Start with the linear kernel function but explore all of the function in lab instruction section 3.3
DONE 2. Define objective (a function which takes the α-vector as argument and returns a scalar value) 
   This function should effectively implement equation 4 in the lab instructions. NOTE THAT THIS FUNCTION WILL BE CALLED
   SEVERAL HUNDRED TIMES, SO MAKE IT EFFICIENT. Define P as a global numpy array
DONE 3. Define zerofun (zerofun is a function you have defined which calculates the value which
   should be constrained to zero. Like objective, zerofun takes a vector as
   argument and returns a scalar value. 
DONE 4. Define a function that creates the matrix P from the data points  
5. Call minimize
6. Extrac the non-zero α values (use 10^-5 as the limit). Save non zero α values with the corresponding data points xi and
   target values ti in a separate data structure, for example a list.
7. Calculate b (the bias) using equation 7 in lab instruction
8. Implement the indicator function ind(s). ((equation 6) which uses the non-zero α values together with the corresponding
   xi and ti to classify new points.
9. Generate test data according to the lab instructions section 5
10. Plot the data and the decision boundary according to section 6
11. After completing above tasks with the linear kernel function, move on to questions under section 7.
""" 

def kernel(x, y, kernel_func, p = 1, sigma = 1):
    if (kernel_func == "linear"):
        scalar = numpy.dot(x, y)
    elif (kernel_func == "poly"):
        scalar = (numpy.dot(x, y) + 1) ** p
    elif(kernel_func == "rbf"):
        scalar = math.exp(-(numpy.linalg.norm(x-y) ** 2) / (2 * sigma ** 2))
    return scalar

def make_matrix():
    global P
    P = []
    for i in range(len(targets)):
        P.append([target[i]*target[j]*kernel(inputs[i], inputs[j], "linear") for j in range(len(targets))])

## Implementing equation 4
## Should return a scalar
def objective(vector_alpha):
    scalar = 0.5 * numpy.sum(numpy.multiply(numpy.outer(vector_alpha, vector_alpha), P)) - numpy.sum(vector_alpha)
    return scalar

def zerofun(vector_alpha):
    scalar = numpy.dot(vector_alpha, targets)
    return scalar

def genData():
    numpy.random.seed(100) # TODO: Comment out
    classA = numpy.concatenate((numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5], numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
    classB = numpy.random.randn(20, 2) * 0.2 + [0.0 , -0.5]
    inputs = numpy.concatenate((classA , classB))
    targets = numpy.concatenate((numpy.ones(classA.shape[0]), -numpy.ones(classB.shape[0])))
    N = inputs.shape[0] # Number of rows (samples)
    permute = list(range(N)) 
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    
def main():
    global targets, inputs, N
    targets = None
    inputs = None
    N = None
    genData()
    make_matrix()

    # B = #list of pairs stating the lower and upper bounds. same length as alpha.
    # XC={'type':'eq', 'fun':zerofun}
    # start = numpy.zeros(N) # N = number of training samples
    # # ret = minimize( objective , start ,bounds=B, constraints=XC)
    # # alpha = ret [ ’x ’ ]

    print(start)

main()
