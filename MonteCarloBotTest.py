#played as v45

import chess
import chess.engine
import chess.syzygy
import random
import math
import numpy as np
from collections import defaultdict
from reconchess import *
import os


import operator
from collections import Counter
from itertools import groupby
import chess.svg

STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'
os.environ['STOCKFISH_EXECUTABLE'] = r"C:\Users\kimbe\Documents\Brady Stuff\NRL Stuff\Chess Tools\lc0-v0.28.2-windows-cpu-dnnl\lc0.exe"


tablebase = chess.syzygy.open_tablebase(r"C:\Users\kimbe\Documents\Brady Stuff\NRL Stuff\Chess Tools\syzygy")
#self.tablebase.add_directory(r"E:\SEAP 2021\Syzygy-6")
# make sure stockfish environment variable exists
if STOCKFISH_ENV_VAR not in os.environ:
    raise KeyError(
        'TroutBot requires an environment variable called "{}" pointing to the Stockfish executable'.format(
            STOCKFISH_ENV_VAR))

# make sure there is actually a file
stockfish_path = os.environ[STOCKFISH_ENV_VAR]
if not os.path.exists(stockfish_path):
    raise ValueError('No stockfish executable found at "{}"'.format(stockfish_path))

# initialize the stockfish engine
global engine
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
engine.configure({"SyzygyPath":r"C:\Users\kimbe\Documents\Brady Stuff\NRL Stuff\Chess Tools\syzygy"})


# MOVEMENT SOIMCTS
    
def MovementSOISMCTS(board_set, color, returnNode = False):
    root = MovementMonteCarloTreeSearchNode(board_set = board_set, root_node = True, color = color)
    selected_nodes = root.best_action()
    
    if returnNode:
        return root
    return selected_nodes
        
    """
    make a tree
    (v,d) = select a node from starting state
    
    if there are still actions at this node:
        (v,d) = new node made by expanding current (v,d)
        
    get reward by simulating board
    backpropogate that reward
    
    return an action
        
    """       

    

class MovementMonteCarloTreeSearchNode():
    def __init__(self, state = None, parent=None, parent_action=None, root_node = False, board_set = None, color = None):
        self.state = state
        #print(self.state)
        self.board_set = board_set
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self.availability = 0 
        self._results = defaultdict(int)
        self.reward = 0
        self._untried_actions = {}
        self._actions = {}
        self.root_node = root_node
        self.color = color
        """
        stockfish_path = os.environ[STOCKFISH_ENV_VAR]
        if not os.path.exists(stockfish_path):
            raise ValueError('No stockfish executable found at "{}"'.format(stockfish_path))
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        """
        if not self.root_node:
            self._untried_actions = []
            self._untried_actions = self.untried_actions()
            self._actions = []
            self._actions = self.actions()
        
    
    
    def untried_actions(self):
        self._untried_actions = self.get_legal_actions(self.state)
        return self._untried_actions
    
    def actions(self):
        self._actions = self.get_legal_actions(self.state)
        return self._actions
    
    def get_visits(self):
        return self._number_of_visits
    
    
    def get_availability(self):
        return self.availability
    
    def get_reward(self):
        return self.reward
    
    
    def expand(self):
        if self.root_node:
            #print("Root node children: " + str(len(self.children)))
            
            action = self._untried_actions[self.state.fen()].pop(random.randint(0,len(self._untried_actions[self.state.fen()])-1))

            
        else:
            #print("Other node children: " + str(len(self.children)))
            action = self._untried_actions.pop(random.randint(0,len(self._untried_actions)-1))
            
        next_state = self.move(self.state, action)
        #print(next_state)
        child_node = MovementMonteCarloTreeSearchNode(state=next_state, parent=self, parent_action=action, color = self.color)
    
        self.children.append(child_node)
        return child_node
    
        """
        //expansion
        if v is nonterminal then
            choose at random an action a from node v that is compatible with d and does not exist in the tree
            add a child node to corresponding to the player 1 information set reached using action a
            and set it as the new current node v
        """
    
    
    def is_terminal_node(self):
        return self.is_game_over(self.state)
    
    
    
    def simulate(self):
        new_board = self.state.copy()
        new_board.clear_stack()
        
        x = self.singleRandomSim(new_board)
        #print(x)
        return x
    
    def singleRandomSim(self, board):
        #print("random sim")
        new_board = board.copy()
        color = new_board.turn
        while (not self.is_game_over(new_board)):
            enemy_king_square = new_board.king(not new_board.turn)

            try:
                enemy_king_attackers = new_board.attackers(new_board.turn, enemy_king_square)
            except TypeError:
                #print("Type Error Trapped")
                enemy_king_attackers = False

            if enemy_king_attackers:
                #print("Opponent king open!")
                attacker_square = enemy_king_attackers.pop()
                best_move = chess.Move(attacker_square, enemy_king_square)

            else:
                best_move = random.choice(self.get_legal_actions(new_board))

            new_board.push(best_move)

        return self.game_result(new_board, color)
    
    def simulate_2(self):
        
        board = self.state
        new_board = board.copy()
        new_board.clear_stack()
        return self.evaluate_board(new_board, self.color)
    
    def evaluate_board(self, board, color):
        
        """
        Checks if game has ended
        """
        if self.is_game_over(board):
            if self.game_result(board,color) < 1:
                return -9999
            else:
                return 9999
        
        """
        Checks enemy king square
        """
        enemy_king_square = board.king(not color)
        
        try:
            enemy_king_attackers = board.attackers(color, enemy_king_square)
        except TypeError:
            #print("Type Error Trapped")
            enemy_king_attackers = False
        
        if enemy_king_attackers:
            opponent_vulnerable = 5000
            
        else:
            opponent_vulnerable = 0
        
        """
        Checks my king square
        """
        my_king_square = board.king(color)
        
        try:
            my_king_attackers = board.attackers(not color, my_king_square)
        except TypeError:
            #print("Type Error Trapped")
            my_king_attackers = False
        
        if my_king_attackers:
            #print("king being attacked!")
            me_vulnerable = 4000
            
        else:
            me_vulnerable = 0
            
        
        
        wp = len(board.pieces(chess.PAWN, chess.WHITE))
        bp = len(board.pieces(chess.PAWN, chess.BLACK))
        wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
        bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
        wb = len(board.pieces(chess.BISHOP, chess.WHITE))
        bb = len(board.pieces(chess.BISHOP, chess.BLACK))
        wr = len(board.pieces(chess.ROOK, chess.WHITE))
        br = len(board.pieces(chess.ROOK, chess.BLACK))
        wq = len(board.pieces(chess.QUEEN, chess.WHITE))
        bq = len(board.pieces(chess.QUEEN, chess.BLACK))
            
        material = 100*(wp-bp)+320*(wn-bn)+330*(wb-bb)+500*(wr-br)+900*(wq-bq)
        
        
        
        pawntable = [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10,-20,-20, 10, 10,  5,
         5, -5,-10,  0,  0,-10, -5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5,  5, 10, 25, 25, 10,  5,  5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
         0,  0,  0,  0,  0,  0,  0,  0]

        knightstable = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50]

        bishopstable = [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10,-10,-10,-10,-10,-20]

        rookstable = [
          0,  0,  0,  5,  5,  0,  0,  0,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
          5, 10, 10, 10, 10, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0]

        queenstable = [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  5,  5,  5,  5,  5,  0,-10,
          0,  0,  5,  5,  5,  5,  0, -5,
         -5,  0,  5,  5,  5,  5,  0, -5,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20]

        kingstable = [
         20, 30, 10,  0,  0, 10, 30, 20,
         20, 20,  0,  0,  0,  0, 20, 20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30]
        
        
        pawnsq = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
        pawnsq= pawnsq + sum([-pawntable[chess.square_mirror(i)] 
                                        for i in board.pieces(chess.PAWN, chess.BLACK)])
        knightsq = sum([knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
        knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)] 
                                        for i in board.pieces(chess.KNIGHT, chess.BLACK)])
        bishopsq= sum([bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
        bishopsq= bishopsq + sum([-bishopstable[chess.square_mirror(i)] 
                                        for i in board.pieces(chess.BISHOP, chess.BLACK)])
        rooksq = sum([rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)]) 
        rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)] 
                                        for i in board.pieces(chess.ROOK, chess.BLACK)])
        queensq = sum([queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)]) 
        queensq = queensq + sum([-queenstable[chess.square_mirror(i)] 
                                        for i in board.pieces(chess.QUEEN, chess.BLACK)])
        kingsq = sum([kingstable[i] for i in board.pieces(chess.KING, chess.WHITE)]) 
        kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)] 
                                        for i in board.pieces(chess.KING, chess.BLACK)])
        
        evaluation = material + pawnsq + knightsq + bishopsq+ rooksq+ queensq + kingsq 
        if board.turn == color:
            return evaluation + opponent_vulnerable - me_vulnerable
        else:
            return -evaluation + opponent_vulnerable - me_vulnerable
    
    
    
    
    
    
    def simulate_3(self):
        new_board = self.state.copy()
        new_board.clear_stack()
        
        
        
        enemy_king_square = new_board.king(not new_board.turn)
        my_king_square = new_board.king(new_board.turn)
        try:
            enemy_king_attackers = new_board.attackers(new_board.turn, enemy_king_square)
        except TypeError:
            #print("Type Error Trapped")
            enemy_king_attackers = False
            
        try:
            my_king_attackers = new_board.attackers(not new_board.turn, my_king_square)
        except TypeError:
            #print("Type Error Trapped")
            my_king_attackers = False
            
        if enemy_king_attackers:
            #print("I can win")
            return 1000
        elif my_king_attackers:
            #print("I can lose")
            return -1000
        else:
        
        
            info = engine.analyse(new_board, chess.engine.Limit(time=0.1))
            return info['score'].wdl().pov(self.state.turn).wins - info['score'].wdl().pov(self.state.turn).losses
    
    
    
        """
        //simulation
        run a simulation to the end of the game using determinization d
        """
    
    
    def backpropagate(self, result):
        self._number_of_visits += 1.
        self.reward += result
        if self.parent:
            self.parent.backpropagate(result)
            
            for child in self.parent.children:
                child.availability += 1
            
        """
        //backpropogation
        for each node visited during this iteration do
        
            update u’s visit count and total simulation reward
            
            for each sibling w of u that was available for
            selection when u was selected, including u itself do
                update w’s availability count
        """
    
    def is_fully_expanded(self):
        
        if self.root_node:
            return len(self._untried_actions[self.state.fen()]) == 0
        
        else:
            return len(self._untried_actions) == 0
    
    
    def best_child(self, exploration=0.7, backup = False, printResult = False):
        #print("best child call")
        choices_weights = []
        for c in self.children:
            reward = c.get_reward()
            #print(reward)
            visits = c.get_visits()
            #print(visits)
            availability = c.get_availability()
            #choices_weights.append(reward / visits)
            choices_weights.append(reward / visits + exploration * math.sqrt(math.log(availability)/visits))
           # print(choices_weights)
           
           
        if printResult:
            #print(self.children[np.argmax(choices_weights)].parent_action, self.children[np.argpartition(choices_weights, -1)[-2]].parent_action)
            print([c.parent_action for c in self.children])
            print(choices_weights)
            #print(np.argmax(choices_weights))
            
            
        if backup:
            
            return self.children , choices_weights
            #return self.children[np.argmax(choices_weights)], self.children[np.argpartition(choices_weights, -1)[-2]]
        else:
            return self.children[np.argmin(choices_weights)]
    
    
    
    def select_and_expand(self):
        
        current_node = self
        #print(len(current_node.children))    
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node  
        
        """
        //Selection
        repeat
            descend the tree (restricted to nodes/actions compatible with d) using the chosen bandit algorithm 
            
        until a node is reached such that some action from v leads to a player 1 information set which is not currently in the tree
        
        or 
        
        until v is terminal
        """
    
   
    def best_action(self):
        exploration_number = 1000
	
        for i in range(exploration_number):
            
            if self.root_node:
                #print(self.state)
                #print("")
                #if self.state == None or len(self._untried_actions[self.state.fen()]) == 0:
                determinization = random.choice(self.board_set)
                #print("determinization: ")
                #print(determinization)
                #print("")
                self.state = determinization
                if self.state.fen() not in self._untried_actions:
                    self._untried_actions[self.state.fen()] = self.get_legal_actions(self.state)
                    self._actions[self.state.fen()] = self.get_legal_actions(self.state)
                
            v = self.select_and_expand()
            #print("simulated board: ")
            #print(v.state)
            #print("")
            reward = v.simulate()
            v.backpropagate(reward)
	
        return self.best_child(backup=True, printResult = False)
    
    
    
    def get_legal_actions(self, board): 
        return list(board.pseudo_legal_moves)
        
    def is_game_over(self, board):
        return board.king(True) == None or board.king(False) == None
        
    def game_result(self, board, color):
        
        if board.king(color) == None:
            x = -1
            return x
        if board.king(not color) == None:
            x = +1
            return x
        
    def move(self, board,action):
        new_board = board.copy()
        new_board.push(action)
        return new_board


# SENSING SOISMCTS

def SensingSOISMCTS(board_set, color):
    root = SensingMonteCarloTreeSearchNode(board_set = board_set, root_node = True, color = color)
    selected_node = root.best_action()
    return selected_node
        
    """
    make a tree
    (v,d) = select a node from starting state
    
    if there are still actions at this node:
        (v,d) = new node made by expanding current (v,d)
        
    get reward by simulating board
    backpropogate that reward
    
    return an action
        
    """       

    

class SensingMonteCarloTreeSearchNode():
    def __init__(self, state = None, parent=None, parent_action=None, root_node = False, board_set = None, color = None):
        self.state = state
        #print(self.state)
        self.board_set = board_set
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self.availability = 0 
        self._results = defaultdict(int)
        self.reward = 0
        self._untried_actions = {}
        self._actions = {}
        self.root_node = root_node
        self.color = color
        
        if not self.root_node:
            self._untried_actions = []
            self._untried_actions = self.untried_actions()
            self._actions = []
            self._actions = self.actions()
    
    
    
    def untried_actions(self):
        self._untried_actions = self.get_legal_actions(self.state)
        return self._untried_actions
    
    def actions(self):
        self._actions = self.get_legal_actions(self.state)
        return self._actions
    
    def get_visits(self):
        return self._number_of_visits
    
    
    def get_availability(self):
        return self.availability
    
    def get_reward(self):
        return self.reward
    
    
    def expand(self):
        #print(self._untried_actions)
        
        if self.root_node:
            if len(self._untried_actions[self.state.fen()]) != 0:
                action = self._untried_actions[self.state.fen()].pop(random.randint(0,len(self._untried_actions[self.state.fen()])-1))
            else:
                action = random.choice(self._actions[self.state.fen()])
            
        else:
            action = self._untried_actions.pop(random.randint(0,len(self._untried_actions)-1))
        #print(action)
        next_board_set = self.move(self.board_set,self.state, action)
        #print(next_state)
        child_node = SensingMonteCarloTreeSearchNode(state=self.state, parent=self, parent_action=action, board_set = next_board_set, color = self.color)
    
        self.children.append(child_node)
        return child_node
    
        """
        //expansion
        if v is nonterminal then
            choose at random an action a from node v that is compatible with d and does not exist in the tree
            add a child node to corresponding to the player 1 information set reached using action a
            and set it as the new current node v
        """
    
    
    def is_terminal_node(self):
        return self.is_game_over(self.board_set)
    
    
    def simulate(self):

        return self.game_result(self.board_set)

        
    
        """
        //simulation
        run a simulation to the end of the game using determinization d
        """
    
    
    def backpropagate(self, result):
        self._number_of_visits += 1.
        self.reward += result
        if self.parent:
            self.parent.backpropagate(result)
            
            for child in self.parent.children:
                child.availability += 1
            
        """
        //backpropogation
        for each node visited during this iteration do
        
            update u’s visit count and total simulation reward
            
            for each sibling w of u that was available for
            selection when u was selected, including u itself do
                update w’s availability count
        """
    
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    
    
    def best_child(self, exploration=0.7, backup = False, printResult = False):
        #print("best child call")
        choices_weights = []
        for c in self.children:
            reward = c.get_reward()
            #print(reward)
            visits = c.get_visits()
            #print(visits)
            availability = c.get_availability()
            #choices_weights.append(reward / visits)
            choices_weights.append(reward / visits + exploration * math.sqrt(math.log(availability)/visits))
           # print(choices_weights)
           
        if backup:
            if printResult:
                #print(self.children[np.argmax(choices_weights)].parent_action, self.children[np.argpartition(choices_weights, -1)[-2]].parent_action)
                print([c.parent_action for c in self.children])
                print(choices_weights)
                #print(np.argmax(choices_weights))
            
            return self.children , choices_weights
            #return self.children[np.argmax(choices_weights)], self.children[np.argpartition(choices_weights, -1)[-2]]
            
        else:
            return self.children[np.argmax(choices_weights)]
    
    
    
    def select_and_expand(self):
        
        current_node = self
        #print(current_node.children)    
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node  
        
        """
        //Selection
        repeat
            descend the tree (restricted to nodes/actions compatible with d) using the chosen bandit algorithm 
            
        until a node is reached such that some action from v leads to a player 1 information set which is not currently in the tree
        
        or 
        
        until v is terminal
        """
    
   
    def best_action(self):
        exploration_number = 1000
	
        for i in range(exploration_number):
            
            if self.root_node:
                determinization = random.choice(self.board_set)
                self.state = determinization
                #print(self.board_set)
                #print(self.state)
                if self.state.fen() not in self._untried_actions:
                    self._untried_actions[self.state.fen()] = self.get_legal_actions(self.state)
                    self._actions[self.state.fen()] = self.get_legal_actions(self.state)
                
            v = self.select_and_expand()
            reward = v.simulate()
            v.backpropagate(reward)
	
        return self.best_child(backup=True)
    
    
    
    def get_legal_actions(self, state): 
        return [9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 33, 34,35,36, 37, 38, 41, 42, 43, 44, 45, 46, 49, 50, 51, 52, 53, 54]
        
    def is_game_over(self, board_set):
        return len(board_set) == 1
        
    def game_result(self, board_set):
        if len(board_set) <= 10:
            return 1
        
        if len(board_set) > 10:
            return -1
        
    def move(self, board_set, determinization, action):
        new_board_set = board_set.copy()
        
        sense_squares = self.sense_at(action)
        piece_map = determinization.piece_map()
        sense_result = []
        
        for square in sense_squares:
            if square in piece_map.keys():
                sense_result.append((square, piece_map[square]))
            else:
                sense_result.append((square, None))
        
        for board in new_board_set:
            possible = True
            for square, piece in sense_result:
                if board.piece_at(square) != piece:
                    possible = False
            if not possible:
                new_board_set.remove(board)
                     
        return new_board_set

    def sense_at(self, square_num):
        square_str = chess.SQUARE_NAMES[square_num]
        rank_str = square_str[0]
        rank_int = ord(rank_str)
        sense_ranks = [chr(rank_int - 1), chr(rank_int) , chr(rank_int + 1)]
        
        file_str = square_str[1]
        file_int = int(file_str)
        sense_files = [str(file_int - 1), str(file_int), str(file_int + 1)]
        
        sense_squares = []
        for rank in sense_ranks:
            for file in sense_files:
                sense_squares.append(rank + file)
                
        int_sense_squares = []
        for square in sense_squares:
            int_sense_squares.append(chess.parse_square(square))
                
        return int_sense_squares
 
#MHT HELPER FUNCTION

def mode(l):
    freqs = groupby(Counter(l).most_common(), lambda x:x[1])
    x = [val for val,count in next(freqs)[1]]
    
    try:
        y = [val for val,count in next(freqs)[1]]
        
    except StopIteration:
        y = []
    
    return x,y
       

# RECONCHESS CLASS

class MonteCarloBot(Player):
    
    
    def __init__(self):
        #print("HELLO WORLD")
        self.board = None
        self.color = None
        self.opponent_color = None
        self.board_set = []
        self.my_piece_captured_square = None
        self.need_new_boards = True
        self.sense_dict = {0:9,1:9,2:10,3:11,4:12,5:13,6:14,7:14,8:9,15:14,16:17,23:22,24:25,31:30,32:33,39:38,40:41,47:46,48:49,55:54,56:49,57:49,58:50,59:51,60:52,61:53,62:54,63:54}
        self.piece_scores = {'P':1, 'N':3, 'B':3, 'R':5, 'Q':9, 'K':9}
        self.MHT_Sense = False
        self.MHT_Move = False
        self.random_sampling = True
        
        
        self.tablebase = chess.syzygy.open_tablebase(r"C:\Users\kimbe\Documents\Brady Stuff\NRL Stuff\Chess Tools\syzygy")
        #self.tablebase.add_directory(r"E:\SEAP 2021\Syzygy-6")
        # make sure stockfish environment variable exists
        if STOCKFISH_ENV_VAR not in os.environ:
            raise KeyError(
                'TroutBot requires an environment variable called "{}" pointing to the Stockfish executable'.format(
                    STOCKFISH_ENV_VAR))

        # make sure there is actually a file
        stockfish_path = os.environ[STOCKFISH_ENV_VAR]
        if not os.path.exists(stockfish_path):
            raise ValueError('No stockfish executable found at "{}"'.format(stockfish_path))

        # initialize the stockfish engine
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({"SyzygyPath":r"C:\Users\kimbe\Documents\Brady Stuff\NRL Stuff\Chess Tools\syzygy"})
    
    
    def evaluate_board(self, board, color):
            
            """
            Checks if game has ended
            """
            if self.is_game_over(board):
                if self.game_result(board,color) < 1:
                    return -9999
                else:
                    return 9999
            
            """
            Checks enemy king square
            """
            enemy_king_square = board.king(not color)
            
            try:
                enemy_king_attackers = board.attackers(color, enemy_king_square)
            except TypeError:
                #print("Type Error Trapped")
                enemy_king_attackers = False
            
            if enemy_king_attackers:
                opponent_vulnerable = 5000
                
            else:
                opponent_vulnerable = 0
            
            """
            Checks my king square
            """
            my_king_square = board.king(color)
            
            try:
                my_king_attackers = board.attackers(not color, my_king_square)
            except TypeError:
                #print("Type Error Trapped")
                my_king_attackers = False
            
            if my_king_attackers:
                #print("king being attacked!")
                me_vulnerable = 4000
                
            else:
                me_vulnerable = 0
                
            
            
            wp = len(board.pieces(chess.PAWN, chess.WHITE))
            bp = len(board.pieces(chess.PAWN, chess.BLACK))
            wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
            bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
            wb = len(board.pieces(chess.BISHOP, chess.WHITE))
            bb = len(board.pieces(chess.BISHOP, chess.BLACK))
            wr = len(board.pieces(chess.ROOK, chess.WHITE))
            br = len(board.pieces(chess.ROOK, chess.BLACK))
            wq = len(board.pieces(chess.QUEEN, chess.WHITE))
            bq = len(board.pieces(chess.QUEEN, chess.BLACK))
                
            material = 100*(wp-bp)+320*(wn-bn)+330*(wb-bb)+500*(wr-br)+900*(wq-bq)
            
            
            
            pawntable = [
             0,  0,  0,  0,  0,  0,  0,  0,
             5, 10, 10,-20,-20, 10, 10,  5,
             5, -5,-10,  0,  0,-10, -5,  5,
             0,  0,  0, 20, 20,  0,  0,  0,
             5,  5, 10, 25, 25, 10,  5,  5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
             0,  0,  0,  0,  0,  0,  0,  0]
    
            knightstable = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50]
    
            bishopstable = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -20,-10,-10,-10,-10,-10,-10,-20]
    
            rookstable = [
              0,  0,  0,  5,  5,  0,  0,  0,
             -5,  0,  0,  0,  0,  0,  0, -5,
             -5,  0,  0,  0,  0,  0,  0, -5,
             -5,  0,  0,  0,  0,  0,  0, -5,
             -5,  0,  0,  0,  0,  0,  0, -5,
             -5,  0,  0,  0,  0,  0,  0, -5,
              5, 10, 10, 10, 10, 10, 10,  5,
             0,  0,  0,  0,  0,  0,  0,  0]
    
            queenstable = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  5,  5,  5,  5,  5,  0,-10,
              0,  0,  5,  5,  5,  5,  0, -5,
             -5,  0,  5,  5,  5,  5,  0, -5,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20]
    
            kingstable = [
             20, 30, 10,  0,  0, 10, 30, 20,
             20, 20,  0,  0,  0,  0, 20, 20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30]
            
            
            pawnsq = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
            pawnsq= pawnsq + sum([-pawntable[chess.square_mirror(i)] 
                                            for i in board.pieces(chess.PAWN, chess.BLACK)])
            knightsq = sum([knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
            knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)] 
                                            for i in board.pieces(chess.KNIGHT, chess.BLACK)])
            bishopsq= sum([bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
            bishopsq= bishopsq + sum([-bishopstable[chess.square_mirror(i)] 
                                            for i in board.pieces(chess.BISHOP, chess.BLACK)])
            rooksq = sum([rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)]) 
            rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)] 
                                            for i in board.pieces(chess.ROOK, chess.BLACK)])
            queensq = sum([queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)]) 
            queensq = queensq + sum([-queenstable[chess.square_mirror(i)] 
                                            for i in board.pieces(chess.QUEEN, chess.BLACK)])
            kingsq = sum([kingstable[i] for i in board.pieces(chess.KING, chess.WHITE)]) 
            kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)] 
                                            for i in board.pieces(chess.KING, chess.BLACK)])
            
            evaluation = material + pawnsq + knightsq + bishopsq+ rooksq+ queensq + kingsq 
            if board.turn == color:
                return evaluation + opponent_vulnerable - me_vulnerable
            else:
                return -evaluation + opponent_vulnerable - me_vulnerable
            
    def is_game_over(self, board):
        return board.king(True) == None or board.king(False) == None
        
    def game_result(self, board, color):
        
        if board.king(color) == None:
            x = 1
            return x
        if board.king(not color) == None:
            x = -1
            return x
        

    def possibleMoves(self, board, color):
        new_board = board.copy()
        new_board.turn = color
        return list(new_board.pseudo_legal_moves)
    
    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        #print(board)
        self.board_set.append(board)
        self.color = color
        self.opponent_color = not color
        self.first_turn = True
        
       
    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        
        #print("")
        self.my_piece_captured_square = capture_square
        if captured_my_piece:
            print("PIECE CAPTURED!")
            print("original boards: " + str(len(self.board_set)))
            new_board_set = self.board_set.copy()
            for board in self.board_set:
                new_board_set.remove(board)
                square_attackers = board.attackers(self.opponent_color, self.my_piece_captured_square)

                while square_attackers:
                    new_board = board.copy()
                    attacker_square = square_attackers.pop()
                    new_board.push(chess.Move(attacker_square, self.my_piece_captured_square))
                    new_board_set.append(new_board)
               
            print("new boards: " + str(len(new_board_set)))
            self.board_set = new_board_set
            self.need_new_boards = False
            print("opponent move predict off")

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        if self.MHT_Sense:
            return self.MHT_choose_sense(sense_actions, move_actions, seconds_left)
        
        else:
            return self.MCTS_choose_sense(sense_actions, move_actions, seconds_left)

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        
        
        if self.MHT_Sense:
            self.MHT_handle_sense_result(sense_result)
        else:  
            self.MCTS_handle_sense_result(sense_result)
        
    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        
        
        if self.MHT_Move:
            return self.MHT_choose_move(move_actions, seconds_left)
        else:  
            return self.MCTS_choose_move(move_actions, seconds_left)


    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        #print("MHT move handled")
        
        
        if taken_move != None:
            
            #print("updating boards with my move")
            new_board_set = self.board_set.copy()
            for board in self.board_set:
                new_board_set.remove(board)
                if taken_move in board.pseudo_legal_moves:
                    newboard = board.copy()
                    newboard.push(taken_move)
                    new_board_set.append(newboard)
            
            self.board_set = new_board_set
        
        #else:
            #print("No change, so no move update")
            
        #print("")

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        with open('board_sets.txt', 'a') as file_object:
                file_object.write(str(self.board_set))
    


# MCTS Functions 

    def MCTS_choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        
        
        print("MCTS CHOOSE SENSE")        
        if self.first_turn and self.color:
            print("first turn white automatic sense")
            return 36
        
        if len(self.board_set) == 1:
            return 36
            
            
        x = self.board_set.copy()
        try:
            children, weights = SensingSOISMCTS(x, color = self.color)
        except IndexError:
            
            print("Index Error - Random Sense")
            return random.choice(sense_actions + [None])
        
        sense_choice = children[np.argmax(weights)].parent_action
        if sense_choice in sense_actions:
            print("Sense MCTS successful")
            if sense_choice in self.sense_dict.keys():
                new_sense = self.sense_dict[sense_choice]
                sense_choice = new_sense
            return sense_choice
        
        else:
            i = -2
            x = -1 * len(weights)
            while(i > x):
                #print(i)
                sense_choice = children[np.argpartition(weights, -1)[i]].parent_action
                if sense_choice in sense_actions:
                    print("Backup MCTS successful after " + str(-1 * i) + " tries")
                    if sense_choice in self.sense_dict.keys():
                        new_sense = self.sense_dict[sense_choice]
                        sense_choice = new_sense
                    return sense_choice
                i -= 1
        
        print("Random sense")
        return random.choice(sense_actions + [None])
        
            #print("sense corrected: " + str(sense_choice))
        
        return sense_choice
            

    def MCTS_handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        
        
        print("MCTS SENSE RESULT")   
        
        if self.first_turn and self.color:
            print("WHITE FIRST TURN SENSE HANDLE EXCEPTION")
            self.first_turn = False
        
        else:
            new_board_set = self.board_set.copy()
            
            if self.need_new_boards:
                
                for board in new_board_set:
                    board.turn = self.opponent_color
                
                #print("predicting opponent moves")
                for board in self.board_set:
                    opponent_moves = self.possibleMoves(board, self.opponent_color)
                    for move in opponent_moves:
                        newboard = board.copy()
                        newboard.push(move)
                        new_board_set.append(newboard)
                        
                for board in new_board_set:
                    board.turn = self.color
                    
                self.board_set = new_board_set.copy()
                
            else:
                self.need_new_boards = True
                #print("opponent move predict back on")  
            
            print("expanded boards: " + str(len(self.board_set)))
                
            #print("now narrowing down")
            for board in self.board_set:
                possible = True
                for square, piece in sense_result:
                    if board.piece_at(square) != piece:
                        possible = False
                if not possible:
                    new_board_set.remove(board)
                    
            
            
            
            self.board_set = new_board_set
            
            
            if len(new_board_set) <= 500:
                self.board_set = new_board_set
                    
            elif self.random_sampling:
                #print("narrowed down")
                self.board_set = random.sample(new_board_set,500)
                
            else:
                print("ORDERED NARROW DOWN")
                board_scores = []
                for board in new_board_set:
                    board_scores.append(self.evaluate_board(board, board.turn))
                
                A = new_board_set
                B = board_scores
                A = np.array(A)
                B = np.array(B)
                #print(len(A))
                #print(len(B))
                inds = B.argsort()
                sorted_a = A[inds]
                self.board_set = list(sorted_a)[0:500]
                
            print("narrow down boards: " + str(len(self.board_set)))
            
        
    
    def MCTS_choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        
        
        print("MCTS CHOOSE MOVE")  
        #print("CHOOSING MOVE")
        x = self.board_set.copy()
        
        #return self.children , choices_weights
        #self.children[np.argmax(choices_weights)], self.children[np.argpartition(choices_weights, -1)[-2]]
        
        try:
            children, weights = MovementSOISMCTS(x, color = self.color)

        except IndexError:
            print("Index Error - Random Move")
            return random.choice(move_actions + [None])
        
        
        move_choice = children[np.argmin(weights)].parent_action
        if move_choice in move_actions:
            print("Movement MCTS successful")
            return move_choice
        else:
            i = -2
            x = -1 * len(weights)
            while(i > x):
                #print(i)
                move_choice = children[np.argpartition(weights, 1)[i]].parent_action
                if move_choice in move_actions:
                    print("Backup MCTS successful after " + str(-1 * i) + " tries")
                    return move_choice
                i -= 1
        
        print("Random move")
        return random.choice(move_actions + [None])
    
    
# MHT Functions    
    
    def MHT_choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        
        print("MHT CHOOSE SENSE")  
        #print("CHOOSING SENSE")
        board_locations = {}
        for square in chess.SQUARES:
            board_locations[square] = [[],0]
    
        for board in self.board_set:
            opponent_moves = self.possibleMoves(board, self.opponent_color)
            for move in opponent_moves:
                location = move.to_square
                piece_symbol = str(board.piece_at(move.from_square)).upper()
                piece_score = self.piece_scores[piece_symbol]
                newboard = board.copy()
                newboard.push(move)
                board_locations[location][0].append(newboard)
                board_locations[location][1] += piece_score
    
        sense_weights = {}
        for location in board_locations:
            sense_weights[location] = len(board_locations[location][0]) * board_locations[location][1]
        
        best_val = max(sense_weights.values())
        keys = sense_weights.keys()
        pot_senses = []
        for key in keys:
            if sense_weights[key] == best_val:
                pot_senses.append(key)
        sense_choice = random.choice(pot_senses)
        #print("sense chosen: " + str(sense_choice))
        
        if sense_choice in self.sense_dict.keys():
            new_sense = self.sense_dict[sense_choice]
            sense_choice = new_sense
            #print("sense corrected: " + str(sense_choice))
        
        return sense_choice
        

    def MHT_handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        
        
        print("MHT SENSE RESULT")  
        
        
        if self.first_turn and self.color:
            print("WHITE FIRST TURN SENSE HANDLE EXCEPTION")
            self.first_turn = False
            
        else:
            new_board_set = self.board_set.copy()
            if self.need_new_boards:
                
                for board in new_board_set:
                    board.turn = self.opponent_color
                
                #print("predicting opponent moves")
                for board in self.board_set:
                    opponent_moves = self.possibleMoves(board, self.opponent_color)
                    for move in opponent_moves:
                        newboard = board.copy()
                        newboard.push(move)
                        new_board_set.append(newboard)
                    
                    
                for board in new_board_set:
                    board.turn = self.color    
                
                self.board_set = new_board_set.copy()
                
                
                
                
            else:
                self.need_new_boards = True
                #print("opponent move predict back on")
                
            #print("now narrowing down")
            for board in self.board_set:
                possible = True
                for square, piece in sense_result:
                    if board.piece_at(square) != piece:
                        possible = False
                if not possible:
                    new_board_set.remove(board)
                    
            
            #print("narrow down boards: " + str(len(new_board_set)))
            
            
            if len(new_board_set) <= 500:
                self.board_set = new_board_set
    
            else:
                syzygy_check = True
                for board in new_board_set:
                    if len(board.piece_map()) > 5:
                        syzygy_check = False
                        break
                
                if syzygy_check:
                    #print("syzygy")
                    wdl_set = []
                    for board in new_board_set:
                        wdl_set.append(self.tablebase.probe_wdl(board))
                        
                    mode_wdl = random.choice(mode(wdl_set))
                    
                    dtz_set = []
                    if mode_wdl > 0:
                        for board in new_board_set:
                            evaluation = self.tablebase.probe_dtz(board)
                            if evaluation <= 0:
                                evaluation = 200
                            dtz_set.append([board, evaluation])
                        sorted_dtz_set = sorted(dtz_set, key=operator.itemgetter(1))
                        final_dtz_set = sorted_dtz_set[0:50]
                        
                        final_board_set = []
                        for item in final_dtz_set:
                            final_board_set.append(item[0])
                        
                        self.board_set = final_board_set
                        
                    else:
                        for board in new_board_set:
                            evaluation = self.tablebase.probe_dtz(board)
                            if evaluation == 0:
                                 evaluation = 178
                            dtz_set.append([board, evaluation])
                        sorted_dtz_set = sorted(dtz_set, key=operator.itemgetter(1), reverse = True)
                        final_dtz_set = sorted_dtz_set[0:50]
                        
                        final_board_set = []
                        for item in final_dtz_set:
                            final_board_set.append(item[0])
                            
                        self.board_set = final_board_set
                    
                elif self.random_sampling:
                    #print("narrowed down")
                    self.board_set = random.sample(new_board_set,500)
                    
                else:
                    print("ORDERED NARROW DOWN")
                    board_scores = []
                    for board in new_board_set:
                        board_scores.append(self.evaluate_board(board, board.turn))
                    
                    A = new_board_set
                    B = board_scores
                    A = np.array(A)
                    B = np.array(B)
                    #print(len(A))
                    #print(len(B))
                    inds = B.argsort()
                    sorted_a = A[inds]
                    self.board_set = list(sorted_a)[0:500]
    
                
            print("possible boards: " + str(len(self.board_set)))
        
        
    def MHT_choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        
        print("MHT CHOOSE MOVE")  
        #print("CHOOSING MOVE")
        pot_moves = []

        #num = 0
        for board in self.board_set:
            #num += 1
            #print(num)
            
            board.turn = self.color
            board.clear_stack()
            
            enemy_king_square = board.king(self.opponent_color)
            
            try:
                enemy_king_attackers = board.attackers(self.color, enemy_king_square)
            except TypeError:
                #print("Type Error Trapped")
                enemy_king_attackers = False
            
            if enemy_king_attackers:
                #print("Opponent king open!")
                attacker_square = enemy_king_attackers.pop()
                best_move = chess.Move(attacker_square, enemy_king_square)
            else:
                try:
                    engine_move = self.engine.play(board, chess.engine.Limit(time = 0.01))
                    best_move = engine_move.move
                        
                except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
                    #print("engine crashed...rebooting")
                    self.engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\brady\OneDrive\Documents\SEAP Internship 2021\lc0-v0.28.0-windows-cpu-dnnl\lc0.exe")
                    best_move = random.choice(move_actions + [None])
                    
            pot_moves.append(best_move)
            
        #print("potential move length: " + str(len(pot_moves)))
        
        if len(pot_moves) > 0:
            move_choice_list_first, move_choice_list_second = mode(pot_moves)
            #print("mode selected: " + str(move_choice_list_first))
            #print("backup: " + str(move_choice_list_second))
            
            possible = False
            
            while not possible:
                
                if move_choice_list_first != []:
                    backup = False
                    move_choice = random.choice(move_choice_list_first)
                
                elif move_choice_list_second != []:
                    #print("backup")
                    backup = True
                    move_choice = random.choice(move_choice_list_second)
                    
                else:
                    #print("going random -- backup failed")
                    return random.choice(move_actions + [None])
                
                
                if move_choice in move_actions:
                    #print("move chosen: " + str(move_choice))
                    possible = True
                    break
                
                
                else:
                    #print("tried impossible move")
                    if not backup:
                        move_choice_list_first.remove(move_choice)
                    
                    else:
                        move_choice_list_second.remove(move_choice)
                
        
            
            return move_choice
        
        
        else:
            #print("going random")
            return random.choice(move_actions + [None])
