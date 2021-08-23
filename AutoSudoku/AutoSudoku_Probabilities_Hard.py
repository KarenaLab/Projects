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
            print(" " + ("-" * 25))

        j = 0
        col = " "
        
        while(j < 9):

            if(j%3 == 0):
                col = col + "| "

            get = str(Board[i, j])

            if(get == "0"):
                get = " "
            
            col = col + get + " "
             
            j = j+1

        col = col + "| "

        print(col)
        i = i+1


    print(" " + ("-" * 25))



    
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

    print("")
    print(f"  Board: Fill= {Board_Fill} ({(Board_Fill/81)*100:.1f}%) ")
    print(f"        Empty= {Board_Empty} \n")

    guess = np.linspace(1, 9, num= 9, dtype= int)
    position = np.linspace(0, 8, num= 9, dtype= int)

    number = 1
    while(number <= 9):

        Prob_Table = np.full((9, 9), np.nan)

        # Creating a Shadow of Probabilities known Number=1, Others= 0
        i = 0
        while(i < 9):

            j = 0
            while(j < 9):

                spot = Board[i, j]

                if(spot == number):
                    Prob_Table[i, j] = 1

                if(spot != number and spot > 0):
                    Prob_Table[i, j] = 0


                j = j+1

            i = i+1


        # Adding Restrictions (Row, Col and Square)

        i = 0
        while(i < 9):

            j = 0
            while(j < 9):

                spot = Prob_Table[i, j]

                if(spot == 1):

                    # Row Restrictions
                    Removing = np.delete(position, j).astype(int)

                    for col in Removing:
                        Prob_Table[i, col] = 0


                    # Column Restrictions
                    Removing = np.delete(position, i).astype(int)

                    for row in Removing:
                        Prob_Table[row, j] = 0


                    # Square Restrictions

                    i_start = (i//3)*3
                    i_end = i_start+3

                    j_start = (j//3)*3
                    j_end = j_start+3

                    i_pos = i_start
                                        
                    while(i_pos < i_end):
                        j_pos = j_start                        

                        while(j_pos < j_end):
                            get = Prob_Table[i_pos, j_pos]
                            #print(f"Prob_Table[{i_pos}, {j_pos}] = {get}")
                            
                            if(np.isnan(get)):
                                Prob_Table[i_pos, j_pos] = 0
                                                              
                            j_pos = j_pos+1

                        i_pos = i_pos+1


                j = j+1

            i = i+1

        # Square Probabilities
        Starter = [0, 3, 6]

        for i_start in Starter:

            i_end = i_start+3

            for j_start in Starter:
                
                j_end = j_start+3
                Square = Prob_Table[i_start:i_end, j_start:j_end]

                No_NaN = np.isnan(Square).sum()

                if(No_NaN == 1):

                    pos = np.argwhere(np.isnan(Square))

                    pos_i = i_start + pos[0][0]
                    pos_j = j_start + pos[0][1] 

                    Board[pos_i, pos_j] = number

                    print(f" > Simple: Board[{pos_i}, {pos_j}] = {number}")


        # Incrasing Number

        number = number+1


    New_Board_Fill = np.count_nonzero(Board != 0)
    Solutions = New_Board_Fill - Board_Fill

    if(Solutions == 0 or New_Board_Fill == 81):

        No_Answer = True
        
        print(f"\n  Final Board")
        BoardPlot(Board)

        print(f" > Number of Turns: {Turns}")
        print(f" > Board Filling: {(Board_Fill/81)*100:.1f}% \n")


    Turns = Turns+1
    
    print("\n")

            
# Closing

print(" * \n")


# Sources ---------------------------------------------------------------

# 
#
#

