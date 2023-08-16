import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

def main():
    parser = argparse.ArgumentParser(description='Create a random fractal landscape')
    parser.add_argument("-N", type =int, help ="matrix size", default = 7)
    parser.add_argument("-S", "--sigma", type =float, help = "degree of steepness of the terrain", default = 1)
    parser.add_argument("-m", "--map_file", help = "file name (pdf or png) with the map", default = False)
    parser.add_argument("-s", "--surf_file", help = "file name (pdf or png), with a plot of the plane", default = False)
    parser.add_argument("-c", "--colormap", help = "colormap from matplotlib to use", default = "viridis")
    parser.add_argument("-f", "--matrix_file", help = "saving the matrix to a file", default = False)
    parser.add_argument("-p", "--matrix_partly", help = "fill an already partially filled matrix", default = False)

    args = parser.parse_args()

    matrix = None

    def diamond_step(mac, squares, n):
        # The function calculates the centers of the submatrices (their indices in squares) and inserts the mean value between the four corners + distorts
        distortion = 2**(n)*args.sigma
        for square in squares:
            values = []
            # values - among which we are looking for the average
            for i in square[0]:
                for j in square[1]:
                    values.append(mac[i,j])
            avg = sum(values)/len(values) + distortion*np.random.normal(0,1)
            diff = int((square[0][1] - square[0][0])/2)
            mac[int(square[0][1] - diff), int(square[1][1] - diff)] = avg
        
    def square_step(mac, squares, n):
        # The function calculates the places to be filled by calculating the indexes of the submatrix centers and moving the coordinates up, down, left, right to a distance that depends on the recursive step. Then, for each place, it calculates the average of neighboring values (if they are in the global matrix) and distorts. 
        distortion = 2**(n-1)*args.sigma
        for square in squares:
            diff = int((square[0][1] - square[0][0])/2)
            neighbours = [[-diff, 0], [0, -diff], [diff, 0], [0, diff]]
            center = [int(square[0][1] - diff), int(square[1][1] - diff)]
            targets = [[center[0]-diff, center[1]], [center[0]+diff, center[1]], [center[0], center[1]-diff], [center[0], center[1]+diff]]

            for p, k in targets:
                avg = 0
                amount = 0
                for n_i, n_j in neighbours:
                    start = p+n_i
                    end = k+n_j
                    if (start)>=0 and (start)<len(mac[0]) and (end)>=0 and (end)<len(mac[0]): 
                        avg += mac[start, end]
                        amount+=1
                mac[p, k] = (avg/amount) + distortion*np.random.normal(0,1)
        
    def main_loop(A, n, divider):
        # The function runs from N to 1. On each iteration, it calculates the indices of the beginnings and ends of the (square) submatrices, on which it performs diamond and square step operations
        length = A.shape[0]-1
        for i in range(n, 0, -1):
            distance = int(length/divider)
            start_end_list = []
            p = 0
            for _ in range(divider):
                start_end_list.append([p, p+distance])
                p+=distance

            squares = []
            for i_p, i_k in start_end_list:
                for j_p, j_k in start_end_list:
                    squares.append([[i_p, i_k], [j_p, j_k]])
            diamond_step(A, squares, i)
            square_step(A, squares, i) 
            divider *= 2

    if args.matrix_partly:
        # Loading a matrix already partially filled and randomizing only from step K. The divisor indicates a recursive step, because we 'divide' the main matrix into a divisor * divisor of the submatrix, and then we increase the divisor
        matrix = np.load(args.matrix_partly, allow_pickle=True)
        divider = 1
        N = int(math.log(matrix.shape[0] -1, 2))
        step = int(math.log(matrix.shape[0] -1, 2))+1

        # Checking if the diamond_step of step 1 has already been performed. If so, we check next steps
        for _ in range(args.N):
            if matrix[int((matrix.shape[0]-1)/divider)][int((matrix.shape[0]-1)/divider)] != 0:
                divider *=2
                step-=1
            else:
                break
        # When the loop finds an unfilled spot, call the main loop with the presented divider 
        main_loop(matrix, step, divider)
    else:
    # start creating the matrix from scratch: set random corner values
        N = args.N
        matrix = np.zeros((2**N+1, 2**N+1), dtype=float)
        for i in [0, 2**N]:
            for j in [0, 2**N]:
                matrix[i,j] = np.random.randint(-20, 20)
        divider = 1
        # call the main loop from step one (divider = 1)
        main_loop(matrix, N, divider)

    if args.matrix_file:
        # save the matrix as matrix_file
        matrix.dump(args.matrix_file)
    
    # Display the map
    plt.imshow(matrix)
    if args.map_file:
        # If filename is given save the map without displaying
        plt.savefig(args.map_file)
    else:
        legend = plt.colorbar(shrink=1)
        plt.show()

    # Display the landscape
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x_data = np.arange(0, 2**N+1, 1)
    y_data = np.arange(0, 2**N+1, 1)
    X, Y = np.meshgrid(x_data, y_data)

    Z = matrix
    c = ax.plot_surface(X, Y, Z, cmap=args.colormap)
    fig.colorbar(c)
    if args.surf_file:
        # If filename is given, save the landscape without displaying
        plt.savefig(args.surf_file)
    else:
        plt.show()
    
if __name__ == "__main__":
    main()
