##	takes in initialized sudoku problem and return solution
##	uses hill climbing approach with small probability of allowing bad moves

import numpy as np
import sys
import random
import matplotlib.pyplot as plt
import time


def initMatrix():
	"""
	initialize and return a 9x9 matrix with 0s
	"""
	matrix = np.zeros((9,9), dtype=int)
	return matrix


def readInput(inputPath):
	"""
	takes in input and return defined matrix and location mask
	location mask for defined cells
	"""
	file = open(inputPath, 'r')
	lines = file.readlines()
	matrix = initMatrix()

	tmp_list = []
	for line in lines:
		line = line.strip()
		tmp_list.append(line.split(" "))
	for idx, line in enumerate(tmp_list):
		if idx > 8:
			print("incorrect input file. Exceed max line")
		if len(line) != 9:
			print("incorrect input file. Row does not contain 9 elements")
		matrix[idx] = line
	
	mask = [ [] for _ in range(9)]
	for idx, line in enumerate(matrix):
		for jdx, num in enumerate(line):
			if num != 0:
				mask[idx].append(jdx)
	
	return matrix, mask


def printMatrix(matrix):
	"""
	takes in defined matrix and output in a nice format
	"""
	for idx, row in enumerate(matrix):
		line = ""
		for jdx, num in enumerate(row):
			if jdx != 0 and jdx%3 == 0:
				line += "| "
			line += str(num) + " "
		print(line)
		if idx == 2 or idx == 5:
			print()


def printMatrixFlat(matrix):
    line = ""
    for idx, num in enumerate(matrix):
        if idx%3 == 0:
            line += "| "
        if idx != 0 and idx%9 == 0:
            line += "\n| "
        if idx != 0 and idx%27 == 0:
            line += "\n| "
        line += str(num) + " "
    line += "|"
    print(line)


def printErrorPlot(error_arr):
    plt.plot(error_arr)
    plt.ylabel("error count")
    plt.xlabel("iteration")
    plt.savefig("errorPlot-"+input_path[:-4]+str(e_probability)+".png")


def hillClimb(matrix, mask, e_prob, step=False, n_step=1):
    """
        find solution with Hill Climbing algo allowing error probability
        can specify how many iterations to take with "n_step" and set "step" to True
        returns solved sudoku and number of iterations taken
        """
    
    # get coordinates and stuff
    coor_t, mask_col = getCoorT(include_masked=True, matrix=matrix)
    coor_box, mask_box = getBoxCoor(include_masked=True, matrix=matrix)
    mask_row = [ [] for _ in range(9)]
    for i, row in enumerate(matrix):
        for num in row:
            if num != 0:
                mask_row[i].append(num)
    
    # fill each row with all numbers
    for idx, row in enumerate(matrix):
        matrix[idx] = fillInArray(row)
    print("fill in numbers...")
    printMatrix(matrix)

    # flatten matrix
    tmp_matrix = []
    for row in matrix:
        tmp_matrix.extend(row)
    matrix = tmp_matrix

    # get initial score
    error_arr = []
    error = checkErrorFlat(matrix, coor_box, coor_t, mask_box, mask_col, mask_row)
    error_arr.append(error)
    error_updated = 0
    solution = getCopyArray(matrix)
    iteration = 0
    tries = 0
    
    bad = random.choices([0,1], weights=(1-e_prob, e_prob), k=1)
    bad = bad[0]
    
    while error != 0:
        
        solution = getCopyArray(matrix)

        # random swap
        col = random.randint(0,8)
        tmp_row = random.sample(range(9),2)
        row1 = tmp_row[0]
        row2 = tmp_row[1]
        
#        # testing another method, look at row ascendingly
#        if row1 == 8:
#            row1 = 0
#        else:
#            row1 += 1

        # only swap if the chosen cell is not predefined
        if col not in mask[row1]:
            # get a random index of the same row to swap
            tmp_list = list(range(9))
            not_mask = list(set(tmp_list).difference(mask[row1]))
            not_mask.remove(col)
            if len(not_mask) > 0:
                rand_col = random.choice(not_mask)
                solution[row1*9:row1*9+9] = swapArray(solution[row1*9:row1*9+9], col, rand_col)
                tries += 1
                if tries > tries_max:
                    print("\nExceeded " + str(tries_max) + " tries in searching.\nProgram terminated with current state as solution.")
                    return matrix, iteration

#            #testing another method
#            # swap two cells from the same col
#            if col not in mask[row2]:
#                if row1 < row2:
#                    solution[row1*9:row2*9+9] = swapArray(matrix[row1*9:row2*9+9], col, (row2-row1)*9+col)
#                else:
#                    solution[row2*9:row1*9+9] = swapArray(matrix[row2*9:row1*9+9], col, (row1-row2)*9)
#            else:
#                continue
        else:
            continue
        
        error_updated = checkErrorFlat(solution, coor_box, coor_t, mask_box, mask_col, mask_row)
        tmp = error_updated-error
        if error >= error_updated or bad == 1:
            tries = 0
            if error < error_updated and debug:
                print("\nmaking a bad move")
                printMatrixFlat(solution)
            matrix = getCopyArray(solution)
            iteration += 1
            if iteration > iter_max:
                print("\nExceeded " + str(iter_max) + " iterations.\nProgram terminated with current state as solution.")
                return matrix, iteration
            error = error_updated
            error_arr.append(error)
            if iteration%50 == 0:
                printErrorPlot(error_arr)
            if debug or iteration%1000 == 0:
                print("\niteration: " + str(iteration))
                print("error: " + str(error))
                printMatrixFlat(matrix)

            if step:
                if n_step <= iteration:
                    printErrorPlot(error_arr)
                    return matrix, iteration
        bad = random.choices([0,1], weights=(1-e_prob, e_prob), k=1)
        bad = bad[0]

    printErrorPlot(error_arr)
    return matrix, iteration
    

def getCopyMatrix(matrix):
    """
    deep copy matrix
    """
    new_matrix = [[] for _ in matrix]
    for i, row in enumerate(matrix):
        for num in row:
            new_matrix[i].append(num)
    return new_matrix
    
    
def getCopyArray(arr):
    """
    deep copy array
    """
    new_array = []
    for num in arr:
        new_array.append(num)
    return new_array
    

def swapArray(arr, i, j):
	"""
	swap element at index i and j of arr
	"""
	tmp = arr[i]
	arr[i] = arr[j]
	arr[j] = tmp
	return arr


def fillInArray(arr):
	"""
	fill in blanks with numbers from 1-9 without repeated number
	"""
	missing = [1,2,3,4,5,6,7,8,9]
	for num in arr:
		if num != 0:
			missing.remove(num)
	random.shuffle(missing)
	for i, num in enumerate(arr):
		if num == 0:
			arr[i] = missing[0]
			missing = missing[1:]
	if missing != []:
		print("leftover number in missing")
		print(missing)
	return arr


def checkRepeat(arr):
	"""
	make sure array has no repeated number
	return a dict of repeated numbers and its index
	"""
	repeat = {}
	seen = []
	for idx, num in enumerate(arr):
		if num not in seen:
			seen.append(num)
		else:
			if num not in repeat:
				repeat[num] = [arr.index(num), idx]
			else:
				repeat[num].append(idx)
	return repeat 


def checkErrorFlat(matrix, coor_box, coor_t, mask_box, mask_col, mask_row):
	"""
	same as checkError but input is flatten
	"""
	error = 0

	# check each column and box
	for i in range(9):
		# check for columns
		repeat = checkRepeat([matrix[idx] for idx in coor_t[i*9:i*9+9]])
#		print(repeat)
		if len(repeat) != 0:
#			print("row " + str(i))
			for num in repeat:
				error += len(repeat[num])-1
#				print("num : " + str(num) + "\tRepeated times: " + str(len(repeat[num])-1) + "\n\t\t*error now: " + str(error))
				if num in mask_col[i]:
					error += (len(repeat[num])-1) *3
#					print("overlapped: " + str(num) + "\n\t\t error now: " + str(error))
		# check for boxes
		repeat = checkRepeat([matrix[idx] for idx in coor_box[i]])
		if len(repeat) != 0:
#			print(str(repeat) + "\nbox: " + str(i))
			for num in repeat:
				error += len(repeat[num])-1
#				print("num : " + str(num) + "\tRepeated times: " + str(len(repeat[num])-1) + "\n\t\t*error now: " + str(error))
				if num in mask_box[i]:
					error += (len(repeat[num])-1) *3
#					print("overlapped: " + str(num) + "\n\t\t error now: " + str(error))
	return error
	

def getBoxCoor(include_masked=False, matrix=None):
	"""
	returns list of coor for each box if matrix is flatten
	also return list of predefined number for each box. Must set include_masked to True and the *original matrix*
	"""
	box_coor = []
	box_pre = [ [] for _ in range(9) ]
	for i in range(9):
		for j in range(9):
			# n-th number in order of box, eg 0-8 are in the first box
			nbox = i//3*9*3 + i%3*3 + j//3*9 + j%3
			box_coor.append(nbox)
			#box_coor.append(i//3*9*3 + i%3*3 + j//3*9 + j%3)
			if include_masked:
				num = matrix[i][j]
				if num != 0:
					box_pre[nbox//9].append(num)
	tmp_coor = [[] for _ in range(9) ]
	for i, nbox in enumerate(box_coor):
		tmp_coor[nbox//9].append(i)
	return tmp_coor, box_pre


def getCoorT(include_masked=False, matrix=None):
	"""
	return list of coor if matrix is transposed and flatten
	also return list of predefined number for each column, must set include_masked to True and the *original matrix*.
	"""
	tmp_coor = [ list(range(j*9, j*9+9)) for j in range(9)]
	tmp_coor = getTranspose(tmp_coor)
	coor = []
	for row in tmp_coor:
		coor.extend(row)
	col_pre = [ [] for _ in range(9) ]
	if include_masked:
		for i, row in enumerate(matrix):
			for j, num in enumerate(row):
				if num != 0:
					col_pre[j].append(num)
	return coor, col_pre


def getTranspose(square_matrix):
	"""
	returns transposed matrix
	"""
	matrixT = [ [] for _ in range(9) ]
	for row in square_matrix:
		for idx, num in enumerate(row):
			matrixT[idx].append(num)
	return matrixT


if __name__ == "__main__":
	print("This is a sudoku solver that uses Hill Climbing algorithm with some error probabilty")
	print("Program will print out matrix every 1000 iteration\n")

	debug = False
	tries_max = 750000
	iter_max = 75000
#	if debug:
#		random.seed(42)	
	
	input_path = sys.argv[1]
	sudoku, location = readInput(input_path)
	print("Sudoku from " + input_path)
	print()
	printMatrix(sudoku)	
	
	try:
		e_probability = float(sys.argv[2])
	except BaseException as e:
		e_probability = 0.001
	print("\nHill Climbing with error probability of " + str(e_probability) + "\n")
	start = time.time()
	#solution, iter = hillClimb(sudoku, location, e_probability, step=True, n_step=50)
	solution, iter = hillClimb(sudoku, location, e_probability)
	end = time.time()
	print("\nSolution:\n")
	printMatrixFlat(solution)
	print("\niterations: " + str(iter))
	print("time taken: " + str(end-start) + "s")
	print("error plot saved as \"errorPlot-" + input_path[:-4] + str(e_probability) + ".png\"")
