# This script file provides fundamental computational functionality for 
# visualizing the Mandelbrot set.

# Import all required Python frameworks.
import os
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# =============================================================================
#                             VARIABELES DECLARATION
# =============================================================================
# Define the bottom left point in our system of coordinates.
START_X, START_Y = -2.0, -1.5 
# Dedine a square area of 3 units is defined relative to our bottom left point.
WIDTH, HEIGHT = 3, 3
# Define the number of pixels per unit.          
DENSITY_PER_UNIT = 250
# Define the real and imaginary axes.           
Real =  np.linspace(START_X, START_X+WIDTH, WIDTH * DENSITY_PER_UNIT)  
Imag = np.linspace(START_Y, START_X+HEIGHT, HEIGHT * DENSITY_PER_UNIT)
# Our lattice will be represented through the utilizatiion of two arrays,  
# namely, Real and Image: the former for the values on the real axis and the  
# latter for the values on the imaginary axis. The number of elements in these 
# two arrays is defined by the variable DENSITY_PER_UNIT which defines the 
# number of samples per unit step. The higher it is, the better quality we get, 
# but at a cost of heavier computation.
# =============================================================================
#                        ADDITIONAL VARIABELES DECLARATION
# =============================================================================
# Set the local figures directory path.
FIGURES_PATH = "figures"
# The following list will be storing the sequence of generated image matrices
IMAGES = [] 
# =============================================================================
#                              FUNCTIONS DEFINITION
# =============================================================================
# Define the main Mandelbrot set computation routine. The following function 
# decides whether a given point in the complex plain pertains to the Mandebrot
# set. We can use the information about the number of iterations before the 
# sequence diverges. All we have to do is to associate this number to a color 
# relative to the maximum number of loops. Thus, for all complex numbers c in 
# some lattice of the complex plane, we can make a nice animation of the 
# convergence process as a function of the maximum allowed iterations.
@jit
def mandelbrot(Re, Im, MaxIterations):
	# This function determines whether the number C = Re + Im * i pertains 
    # to the Mandelbrot set or not. The criterion upon which this decision is 
    # made involves the determination of the convergence status of the 
    # sequence Z[n+1] := Z[n] * Z[n] + C where :
    # Z[0] = Re + Im * i and 
    # Z[n] = X[n] + Y[n] * i.
	# The aforementioned complex series diverges if for some n>=1: 
	#                             2 
    #        |Zn| >= 2  ===>  |Zn|  >= 4 
	#
	
    # Set the initial conditions.
	C = complex(Re, Im)
	Z = complex(Re, Im)
	# Perform the main computation loop.
	for i in range(MaxIterations):
		Z = Z**2 + C
		if abs(Z) > 4.0:
			return i
	return MaxIterations

# Define the main animation routine which plots the Mandebrot set within circle 
# of radius R = 2 within the complex plane. One particularly  interesting 
# area is the 3x3 lattice starting at position -2 and -1.5 for the real and 
# imaginary axis respectively. We can observe the process of convergence as the 
# number of allowed iterations increases. It has to be mentioned that function 
# plays a central role, where the input argument is the frame number, starting 
# from 0. This entails, that in order to animate we always have to think in 
# terms of frames. Hence, we use the frame number to calculate the current 
# number of maximum allowed iterations.
def animate(CurrentIterationIndex,Axes):
    print("Current Iteration: {}".format(CurrentIterationIndex))
    # Clear the axes object.
    Axes.clear()
    # Clear the x-axis ticks.
    Axes.set_xticks([])
    # Clear the y-axis ticks.
    Axes.set_yticks([])

	# Re-initialize the array-like image.
    X = np.empty((len(Real),len(Imag)))

	# Calculate the current number of maximum iterations.
    CurrentMaxIterations = round(1.15**(CurrentIterationIndex+1))

	# Main computation loop for the array-like image for the current number of 
    # maximum iterations. Depending on the current number of maximum iterations, 
    # for every complex number c in our lattice, we calculate the number of 
    # iterations before the sequence diverges. In the end, we interpolate the 
	# values in X and assign them a color drawn from a prearranged colormap.
    for i in range(len(Real)):
        for j in range(len(Imag)):
            X[i, j] = mandelbrot(Real[i], Imag[j], CurrentMaxIterations)

	# Associate colors to the actual iterations performed by the mandelbrot 
    # function with an interpolation.
    img = Axes.imshow(X.T, interpolation="bicubic", cmap="magma")
    return [img]

def image_generator(iteration_indices,figures_path=FIGURES_PATH):
    # Initialize a list object for storing the individual array-like images.
    IMAGES = []
    # Generate the local figures directory.
    os.makedirs(figures_path, exist_ok=True)
    # Loop through the various iteration indices.
    for CurrentIterationIndex in iteration_indices:
        print("Current Image: {}".format(CurrentIterationIndex))
        # Re-initialize the array-like image.
        X = np.empty((len(Real),len(Imag)))
        # Calculate the current number of maximum iterations.
        CurrentMaxIterations = round(1.15**(CurrentIterationIndex+1))
        # Main loop for the computation of the array-like image.
        for i in range(len(Real)):
            for j in range(len(Imag)):
                X[i, j] = mandelbrot(Real[i], Imag[j], CurrentMaxIterations)
        # Append the currenty computed array-like image to the list structure.
        IMAGES.append(X)
        # Set the boundaries for the system of coordinates.
        #x_min, x_max = START_X, START_X+WIDTH
        #y_min, y_max = START_Y, START_Y+HEIGHT
        # Set the name of the current figure.
        figure_name = "mandelbrot_{}".format(CurrentIterationIndex)
        # Set the url of the current figure.
        figure_url = os.path.join(FIGURES_PATH, figure_name)
        # Set the size of the current figure.
        figure_size = (10,10)
        # Create the figure object.
        plt.figure(figsize=figure_size)
        # Remove axis from current figure.
        plt.axis("off")
        # Generate the current image.
        plt.imshow(IMAGES[CurrentIterationIndex].T, cmap ="magma", 
                         interpolation='bicubic')
        # Save the current figure.
        plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
        # Show the current image.
        plt.show()
# =============================================================================
#                                          MAIN  PROGRAM
# =============================================================================
if __name__ == "__main__":
    # Initialize a list of iteration indices and generate the corresponding 
    # sequence of images.
    IterationIndices = range(0,70)
    image_generator(IterationIndices)
    # Instantiate a figure to draw.
    Fig = plt.figure(figsize=(10, 10))
    # Create an axes object.
    Axes = plt.axes()                  
    # Add the axes object within the current image.
    Fig.add_axes(Axes)
    # Perform and save the animation into a new .gif file.
    anim = animation.FuncAnimation(Fig, animate, fargs=(Axes,), frames=70, interval=120, 
                               blit=True)
    anim.save('mandelbrot.gif',writer='pillow')
