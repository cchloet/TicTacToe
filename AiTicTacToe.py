import turtle
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



epsilon = 0.7
learning_rate = 0.7
discount = 0.9

#dif_actions = [1,2,3,4,5,6,7,8,9]

BOARD_PLACES=9
board = [' ' for x in range(9)]

class TicTacToe():
    def __init__(self,player1, player2, Turn= 1):
        self._board = board
        self.player1 = player1 #X
        self.player2 = player2 #O
        self.Turn = Turn
        self.player1.mode = 'idle'
        self.player2.mode = 'idle'
        self.player1.moves = []
        self.player2.moves = []
        self.epsilon = 0.5
        self.player1.random_count = 0
        self.player2.random_count = 0
        self.player1.win_count = 0
        self.player2.win_count = 0
        self.reset()
        self.userWins = 0
    

    def reset(self):
        self._board = [' ' for x in range(9)]
        self.player1.state = (4, 'start')
        self.player2.state = (4, 'start')
        self.Turn = random.choice([0,1])
        self.player1.mode = 'idle'
        self.player2.mode = 'idle'
        self.player1.final_score = 0.0
        self.player2.final_score = 0.0
        self.player1.moves.clear()
        self.player2.moves.clear()
        self.epsilon= 0.7
        self.player1.random_count = 0
        self.player2.random_count = 0

    def winReset(self) :
        self.player1.win_count = 0
        self.player2.win_count = 0
        
    def insertLetter(self, letter, pos):
        self._board[pos] = letter

    def spaceIsFree(self, pos):
        return self._board[pos] == ' '
        

    def printBoard(self):
        print('   |   |')
        print(' ' + self._board[0] + ' | ' + self._board[1] + ' | ' + self._board[2])
        print('   |   |')
        print('-----------')
        print('   |   |')
        print(' ' + self._board[3] + ' | ' + self._board[4] + ' | ' + self._board[5])
        print('   |   |')
        print('-----------')
        print('   |   |')
        print(' ' + self._board[6] + ' | ' + self._board[7] + ' | ' + self._board[8])
        print('   |   |')
        
    def isWinner(self, le):
        return ((self._board[6] == le and self._board[7] == le and self._board[8] == le) or (self._board[3] == le and self._board[4] == le and self._board[5] == le) or(self._board[0] == le and self._board[1] == le and self._board[2] == le) or(self._board[0] == le and self._board[3] == le and self._board[6] == le) or(self._board[1] == le and self._board[4] == le and self._board[7] == le) or(self._board[2] == le and self._board[5] == le and self._board[8] == le) or(self._board[0] == le and self._board[4] == le and self._board[8] == le) or(self._board[2] == le and self._board[4] == le and self._board[6] == le))

    def playerMove(self, move):
        # run = True
        is_valid = self.valid_actions()
        if self.Turn == 0:
            if not (move in is_valid) :
                print('Sorry, this space is occupied!')
                new_mode = 'occupied'
            elif move in is_valid:
                new_mode = 'ran'
                self.insertLetter('X', move)
                self.player1.mode = new_mode
            self.player1.moves.append(move)
            #return self.player1.state(move, new_mode)
        else:
            if not is_valid:
                print('Sorry, this space is occupied!')
                new_mode = 'occupied'
            elif move in is_valid:
                new_mode = 'ran'
                self.insertLetter('O', move)
                self.player2.mode = new_mode
            self.player2.moves.append(move)
            #return self.player2.state(move, new_mode)

    def reward(self):
        if self.Turn == 0:
            curr_mode = self.player1.mode
            if self.isWinner('X'):
                self.player1.win_count += 1
                return 10.00
            elif curr_mode == 'ran':
                return -0.10
            elif curr_mode == 'occupied':
                return -100.00
            else:
                return 0.0
        else:
            curr_mode = self.player2.mode
            if self.isWinner('O'):
                self.player2.win_count += 1
                return 10.00
            elif curr_mode == 'ran':
                return -0.10
            elif curr_mode == 'occupied':
                return -100.00
            else:
                return 0.0

    def move(self, move):
            self.playerMove(move)
            reward = self.reward()
            if self.Turn == 0 :
                self.player1.final_score += reward
                #print("player 1 score: ", self.player1.final_score)
            else :
                self.player2.final_score += reward
                #print("player 2 score: ", self.player2.final_score)
            self.Turn= (self.Turn+1) % 2
            
          # not finished

    def isBoardFull(self):
        if self._board.count(' ') > 0:
            return False
        else:
            return True

    def valid_actions(self):
        actions = [0,1,2,3,4,5,6,7,8]

        for i in actions:
            if not self.spaceIsFree(i):
                actions.remove(i)
        return actions

    def choose_move(self) :
        actions = self.valid_actions()
        choice = random.choice(actions) 
        if random.choice([0.1,1.0]) > self.epsilon :
            if self.Turn == 0 :
                max_value = 0
                cur_max_choice = choice
                for move in actions :
                    cur_choice_val = self.player1.getQval(tuple(self._board), move)
                    if cur_choice_val > max_value :
                        max_value = cur_choice_val
                        cur_max_choice = move
                choice= cur_max_choice
                #print("performing q_value choice...")
            elif self.Turn == 1 :
                max_value = 0
                cur_max_choice = choice
                for move in actions :
                    cur_choice_val = self.player2.getQval(tuple(self._board), move)
                    if cur_choice_val > max_value :
                        max_value = cur_choice_val
                        cur_max_choice = move
                choice= cur_max_choice
                #print("performing q_value choice...")
        #else :
            #print("performing random choice...")
            if self.Turn == 0 :
                self.player1.random_count += 1
            elif self.Turn == 1 :
                self.player2.random_count += 1
        return choice

    def QTrain(self) :
        self.reset()
        games = 2000
        for i in range(games) :
            self.play_game(False)
            self.reset()
            print(i, '/', games)
    
    def play_game(self, show_board) :
        self.reset()
        if(show_board == True) :
            self.epsilon = 0.0
        while not (self.isWinner('X') or self.isWinner('O') or self.isBoardFull()) :
            spot = self.choose_move()
            self.move(spot)
            if(show_board == True) :
                self.printBoard()
                print("")
                print("")
                print("")
            if(self.isWinner('X')):
                print("Player 1 wins!   |Random moves percent for player 1: ", self.player1.random_count / len(self.player1.moves), "|Random moves percent for player 2: ", self.player2.random_count/len(self.player2.moves), "|Player 1 wins: ", self.player1.win_count, "Player 2 wins: ", self.player2.win_count)
            elif(self.isWinner('O')):
                print("Player 2 wins!   |Random moves percent for player 1: ", self.player1.random_count / len(self.player1.moves), "|Random moves percent for player 2: ", self.player2.random_count/len(self.player2.moves), "|Player 1 wins: ", self.player1.win_count, "Player 2 wins: ", self.player2.win_count)
            elif(self.isBoardFull()):
                print("Tie game!   |Random moves percent for player 1: ", self.player1.random_count / len(self.player1.moves), "|Random moves percent for player 2: ", self.player2.random_count/len(self.player2.moves), "|Player 1 wins: ", self.player1.win_count, "Player 2 wins: ", self.player2.win_count)
        player1board= self._board
        player2board = self._board
        #self.printBoard()
        #print("feeding player 1 rewards: ")
        self.player1.feedReward(self.player1.final_score, self.player1.moves ,player1board)
        #print("feeding player 2 rewards: ")
        self.player2.feedReward(self.player2.final_score, self.player2.moves, player2board)

    def userGame(self) :
        self.epsilon = 0.0
        self.reset()
        self.winReset()
        while not (self.isWinner('X') or self.isWinner('O') or self.isBoardFull()) :
            self.printBoard()
            print("")
            print("")
            if self.Turn == 1 : #user first
                playerMove= int(input("Please enter move from 0-8: "))    
                actions = self.valid_actions()
                while playerMove not in actions :
                    playerMove = int(input("Invalid choice, please enter move valid from 0-8: "))
                self.insertLetter('O', playerMove)
                self.Turn= (self.Turn+1) % 2
                print(self.valid_actions())
            else :
                print("Computer's move!")
                spot = self.choose_move()
                self.move(spot)
            if(self.isWinner('X')):
                self.printBoard()
                print("Player 1 wins!   |Player 1 wins: ", self.player1.win_count, "    |Player 2 wins: ", self.userWins)
                self.player1.feedReward(self.player1.final_score, self.player1.moves ,self._board)
                again = input("Play again?(y/n): ")
                if again == 'y' :
                    self.userGame()
                else :
                    self.userWins = 0
            elif(self.isWinner('O')):
                self.userWins +=1
                self.printBoard()
                print("Player 2 wins!   |Player 1 wins: ", self.player1.win_count, "Player 2 wins: ", self.userWins)
                self.player1.feedReward(self.player1.final_score, self.player1.moves ,self._board)
                again = input("Play again?(y/n): ")
                if again == 'y' :
                    self.userGame()
                else :
                    self.userWins = 0
            elif(self.isBoardFull()):
                self.printBoard()
                print("Tie game!   |Player 1 wins: ", self.player1.win_count, "Player 2 wins: ", self.userWins)
                self.player1.feedReward(self.player1.final_score, self.player1.moves ,self._board)
                again = input("Play again?(y/n): ")
                if again == 'y' :
                    self.userGame()
                else :
                    self.userWins = 0

        

class BotPlayer():
    def __init__(self, name):
       self.name= name
       self.q_val = {} #board, move, q_value
       self.Gamma = 0.9
       self.Alpha = 0.7
       
    def feedReward(self, reward, moves, board) :
        #print(moves)
        q_vals = []
        escalating_q_val = []
        minus_rate = reward / len(moves)
        for move in reversed(moves):
            #print("Deleting: ", move)
            board[move] = ' '
            #print("cur q_val: ", q_vals)
            q_vals.append((board, move, reward))
            reward = reward - minus_rate
        for rewards in reversed(q_vals) :
            escalating_q_val.append(rewards)
            #print(board) 
        self.calculateQvals(escalating_q_val)


    def getQval(self, board, action) :
        if self.q_val.get((board, action)) is None:
            self.q_val[(board, action)] = 0.0
        return self.q_val[(board, action)]
    
    def availableActions(self, board) :
        availabilities = []
        for i in range(9) :
            if board[i] == ' ' :
                availabilities.append(i)
        return availabilities
    
    def calculateQvals(self, q_vals) :
        for i in range (len(q_vals)) :
            board, moves, cur_reward = q_vals[i]
            prevQ = self.getQval(tuple(board), moves)
            maxQval = max([self.getQval(tuple(board), cur_move) for cur_move in self.availableActions(board)])
            self.q_val[(tuple(board), moves)] = prevQ + self.Alpha*((cur_reward + self.Gamma * maxQval) - prevQ)

        #Create list of each reward for each move in reverse order, subtracting the reward progressively
    #def getQval(self) :

            




def main():
    player1= BotPlayer('p1')
    player2= BotPlayer('p2')
    print('Welcome to Tic Tac Toe!')
    game1=TicTacToe(player1, player2)
    #game1.printBoard()
    game1.QTrain()
    show_board = False
    print("regular play: ")
    game1.winReset()
    for i in range(40) :
        game1.play_game(show_board)
        print("")
        print("")
        print("")
    game1.userGame()




main()