game = [[1, 0, 0],
[0, 1, 0],
[0, 0, 1]]

def game_board(game_map, player=0, row=0, column=0, display=False):
    try:
        print("   0  1  2 ")
        if not display:
            game[row][column] = player
        for count, row in enumerate(game_map):
            print(count, row)
        return game_map
    except IndexError:
        print("It's broke bro... You put in a weird number somewhere. Everything's 0, 1, or 2.")
        return False
    except Exception as e:
        print(str(e))
        return False

def win(current_game): 
    # horizontal win       
    for row in current_game:
        print(row)
        if row.count(row[0]) == len(row) and row[0] != 0:
            print("You won")
            print(f"Player {row[0]} is the winner")

    # vertical win
    for col in range(len(current_game[0])):
        vertical_check = []

        for row in current_game:
            vertical_check.append(row[0])

        if vertical_check.count(vertical_check[0]) == len(vertical_check) and vertical_check[0] != 0:
            print(f"Player {vertical_check[0]} is the winner")
        
    # \ diagonal win
    diagonal_check = []
    for i in range(len(current_game)):
        diagonal_check.append(current_game[i][i])
    if diagonal_check.count(diagonal_check[0] == len(diagonal_check)) and diagonal_check[0] != 0:
        print(f"Player {diagonal_check[0]} is the winner")
    
    # / diagonal win
    diagonal_check = []
    for i, reverse_i in enumerate(reversed(range(len(current_game)))):
        diagonal_check.append(current_game[i][reverse_i])
    if diagonal_check(diagonal_check[0]) == len(diagonal_check) and diagonal_check[0] != 0:
        print(f"Player {diagonal_check[0]} won")
    

# game_board(game)
# game_board(game, player=1, row=2, column=1)

win(game)
