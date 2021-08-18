# ***  Auto Sudoku  ***
#
#  Project: Auto Soduku
# Filename: AutoSudoku_v01
#  Creator: EKChikui (EKChikui@gmail.com)
#     Date: Aug 12th, 2021
#  Version: 01
#
# -----------------------------------------------------------------------

# Libraries -------------------------------------------------------------

import numpy as np



# Definitions -----------------------------------------------------------

def BoardPlot(Board):

    # Printing the Board with Vertical and Horizontal Separation

    i = 0
    while(i < 9):

        if(i%3 == 0):
            print("-" * 25)

        j = 0
        col = ""
        
        while(j < 9):

            if(j%3 == 0):
                col = col + "| "

            get = str(Board[i, j])

            if(get == "0"):
                get = " "
            
            col = col + get + " "
             
            j = j+1

        col = col + "|"
        print(col)
        i = i+1

    print("-" * 25)


    
# MAIN Program ----------------------------------------------------------

print(f"\n ****  Auto Sudoku Solver (Numpy Edition)  **** \n")

# Starting
Board = np.zeros((9, 9), dtype= int)


# Adding Initial Numbers

while(True):

    get = input(" > Type the Level (e1, e2, m1, m2, h1, x1, x2, x3): ")
    get = get.lower()

    # e= easy, m= medium, h= hard, x= extreme/expert
    # Source: https://sudoku.com/pt/

    if(get == "e1"):

        Board[0, :] = [0, 0, 0, 6, 7, 0, 8, 9, 4]
        Board[1, :] = [0, 0, 9, 8, 3, 1, 5, 0, 0]
        Board[2, :] = [0, 2, 0, 5, 4, 0, 0, 6, 0]
        Board[3, :] = [2, 1, 0, 0, 0, 6, 0, 0, 0]
        Board[4, :] = [4, 0, 6, 0, 0, 7, 3, 8, 0]
        Board[5, :] = [7, 0, 0, 0, 8, 4, 0, 1, 6]
        Board[6, :] = [3, 7, 4, 0, 0, 0, 0, 5, 0]
        Board[7, :] = [0, 0, 5, 0, 0, 3, 2, 0, 0]
        Board[8, :] = [0, 0, 0, 0, 1, 0, 7, 3, 8]

        break


    if(get == "e2"):

        Board[0, :] = [0, 0, 0, 0, 0, 0, 5, 7, 3]
        Board[1, :] = [8, 0, 0, 0, 2, 0, 0, 0, 0]
        Board[2, :] = [7, 0, 0, 9, 0, 0, 8, 1, 0]
        Board[3, :] = [5, 8, 0, 7, 0, 6, 0, 0, 0]
        Board[4, :] = [0, 0, 1, 8, 0, 0, 0, 6, 0]
        Board[5, :] = [2, 3, 0, 0, 4, 0, 0, 0, 9]
        Board[6, :] = [9, 1, 5, 0, 0, 0, 0, 0, 0]
        Board[7, :] = [0, 0, 0, 0, 8, 0, 6, 0, 1]
        Board[8, :] = [0, 0, 0, 0, 0, 0, 0, 4, 0]

        break


    if(get == "m1"):

        Board[0, :] = [0, 0, 0, 0, 0, 6, 0, 3, 0]
        Board[1, :] = [8, 0, 9, 5, 3, 0, 0, 4, 0]
        Board[2, :] = [0, 3, 0, 0, 7, 8, 2, 0, 0]
        Board[3, :] = [0 ,0, 3, 0, 0, 0, 0, 0, 0]
        Board[4, :] = [1, 7, 8, 2, 4, 3, 0, 0, 0]
        Board[5, :] = [0, 0, 2, 0, 0, 0, 3, 0, 4]
        Board[6, :] = [9, 2, 0, 0, 0, 0, 0, 8, 0]
        Board[7, :] = [0, 8, 1, 0, 6, 9, 0, 0, 0]
        Board[8, :] = [0, 0, 0, 8, 0, 1, 7, 9, 0]

        break


    if(get == "m2"):

        Board[0, :] = [9, 0, 0, 6, 0, 1, 7, 0, 0]
        Board[1, :] = [8, 0, 0, 0, 9, 2, 4, 0, 0]
        Board[2, :] = [0, 0, 0, 0, 0, 0, 0, 9, 0]
        Board[3, :] = [0, 0, 0, 2, 0, 0, 8, 5, 4]
        Board[4, :] = [0, 0, 0, 3, 6, 0, 0, 0, 0]
        Board[5, :] = [0, 0, 0, 0, 7, 4, 9, 0, 3]
        Board[6, :] = [2, 9, 0, 4, 0, 0, 0, 0, 0]
        Board[7, :] = [0, 8, 0, 0, 5, 0, 0, 0, 1]
        Board[8, :] = [5, 0, 4, 0, 0, 7, 0, 0, 0]

        break
    

    if(get == "h1"):

        Board[0, :] = [0, 0, 0, 5, 3, 0, 0, 8, 0]
        Board[1, :] = [0, 0, 0, 0, 0, 9, 0, 5, 0]
        Board[2, :] = [0, 4, 0, 8, 0, 7, 0, 1, 0]
        Board[3, :] = [0, 5, 7, 0, 0, 0, 0, 0, 0]
        Board[4, :] = [0, 9, 0, 0, 1, 8, 7, 0, 0]
        Board[5, :] = [2, 0, 0, 0, 5, 0, 1, 0, 0]
        Board[6, :] = [0, 0, 0, 0, 6, 2, 3, 0, 0]
        Board[7, :] = [7, 6, 0, 0, 0, 0, 0, 0, 9]
        Board[8, :] = [0, 0, 0, 0, 0, 4, 0, 0, 0]

        break


    if(get == "x1"):

        Board[0, :] = [0, 4, 0, 0, 0, 0, 1, 0, 0]
        Board[1, :] = [6, 0, 8, 0, 0, 0, 0, 4, 0]
        Board[2, :] = [0, 0, 0, 0, 2, 7, 0, 6, 0]
        Board[3, :] = [0, 9, 7, 0, 0, 0, 0, 0, 0]
        Board[4, :] = [0, 0, 0, 5, 0, 0, 0, 3, 0]
        Board[5, :] = [0, 0, 0, 7, 6, 0, 0, 0, 0]
        Board[6, :] = [0, 0, 2, 0, 0, 9, 0, 5, 0]
        Board[7, :] = [0, 5, 0, 0, 0, 0, 9, 2, 0]
        Board[8, :] = [8, 0, 0, 0, 0, 6, 0, 0, 3]

        break


    if(get == "x2"):

        Board[0, :] = [0, 7, 8, 5, 0, 0, 0, 0, 0]
        Board[1, :] = [0, 0, 3, 0, 0, 7, 8, 0, 0]
        Board[2, :] = [0, 0, 0, 1, 9, 0, 0, 0, 0]
        Board[3, :] = [0, 0, 7, 0, 0, 0, 2, 9, 0]
        Board[4, :] = [0, 9, 0, 0, 6, 1, 0, 4, 0]
        Board[5, :] = [0, 0, 0, 0, 0, 4, 0, 0, 0]
        Board[6, :] = [3, 0, 6, 0, 0, 2, 0, 0, 0]
        Board[7, :] = [0, 1, 0, 0, 0, 0, 0, 0, 4]
        Board[8, :] = [0, 0, 0, 0, 0, 0, 5, 0, 0]

        break


    if(get == "x3"):

        Board[0, :] = [0, 0, 0, 2, 4, 0, 0, 0, 1]
        Board[1, :] = [0, 0, 0, 0, 0, 0, 0, 6, 0]
        Board[2, :] = [3, 6, 0, 0, 0, 0, 5, 7, 4]
        Board[3, :] = [0, 0, 3, 0, 8, 0, 0, 1, 0]
        Board[4, :] = [5, 0, 4, 0, 0, 0, 0, 0, 8]
        Board[5, :] = [0, 0, 0, 7, 0, 0, 0, 0, 0]
        Board[6, :] = [0, 0, 0, 6, 0, 9, 0, 0, 0]
        Board[7, :] = [0, 0, 8, 0, 0, 0, 6, 0, 0]
        Board[8, :] = [0, 7, 0, 0, 0, 4, 0, 9, 2]

        break


    # if... New Board here
    # Copy/Paste = Board[, :] = []


    print(" > Wrong option, type it again \n")



# Starting Info and Check

No_Answer = False

Board_Initial = Board.copy()
Turns = 1

print("\n")


# Square Restrictions

while(No_Answer == False):

    print(f"  Turn {Turns}")
    BoardPlot(Board)

    Board_Empty = np.count_nonzero(Board == 0)
    Board_Fill = np.count_nonzero(Board != 0)
    Solutions = 0

    print(f"\n > Board = Fill: {Board_Fill}, Empty: {Board_Empty}")
    print(f" > Board Filling: {(Board_Fill/81)*100:.1f} % \n")

    guess = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype= int)

    i = 0
    while(i < 9):

        j = 0
        while(j < 9):

            spot = Board[i, j]

            if(spot > 0):

                Prob_Table = np.full((9, 9), np.nan)

                Prob_Table[i, j] = 1
                


            j = j+1


        i = i+1



    # Testing
    No_Answer = True
    
    print("\n")

            
# Closing

print(" * \n")


# Sources ---------------------------------------------------------------

# 


