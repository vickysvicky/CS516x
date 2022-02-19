## How to run

This program is a sudoku solver that is created using Hill Climbing algorithm. The program is written in python and will take in a real Sudoku problem as .txt file while 0’s is used to represent blank spaces and numbers from 1-9 as the problem

The sudoku solver also takes the “bad move” probability as an input, where bad move probability is used here to allow probability for making a bad move. The program will then run iteratively to look for the solution and output both final solution and number of iterations taken.

To run the program with command prompt, go to the directory with both the sudokuSolver.py and an input file. Make sure to have the python packages (numpy, matplotlib) needed for the program to run. 

Run the following command, where ``easy-input.txt`` is the input file name and `0.01` is the bad move probability (if this argument is missing program will default the value to 0.001).

```python sudokuSolver.py easy-input.txt 0.01```

For more information about this program, please see the pdf file that is included in the same directory.
