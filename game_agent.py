"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
from numpy.random import choice
from math import log, sqrt
import json

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Strategy: if there are many spaces available, then chase an opponent
    # otherwise move  opposite direction to opponent.
    player_loc = game.get_player_location(player)
    opponent_loc = game.get_player_location(game.get_opponent(player))
    total_size = game.width * game.height
    blank_space_fraction = float(len(game.get_blank_spaces())) / total_size
    d = distance(player_loc, opponent_loc)
    return (0.5 - blank_space_fraction) * float(d)

def distance(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Strategy: make player "aggressive" and invaide available for opponent moves.
    opponent = game.get_opponent(player)
    opp_moves = game.get_legal_moves(opponent)
    own_moves = game.get_legal_moves(player)
    if game.active_player == player:
        player_loc = game.get_player_location(player)
        for m in opp_moves:
            if m == player_loc:
                return 10
        return float(len(own_moves))
    else:
        opponent_loc = game.get_player_location(opponent)
        for m in own_moves:
            if m == opponent_loc:
                return -10
        return float(len(own_moves))


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Strategy: make player prefer discovering of empty fields and minimise distance of current player to blank spaces
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    player_loc = game.get_player_location(player)
    score = float(own_moves - opp_moves)
    # strategy is to fill diagonal and center crosses
    if player_loc[0] == player_loc[1] or (player_loc[0] % 2 == player_loc[1] % 2):
        score += 0.5
    # if number of moves equal - select with biggest number of moves
    score += 0.2 * own_moves
    return score


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        self.timeout_test()

        moves_scores = dict([(move, self.min_value(game.forecast_move(move), depth - 1)) for move in self.moves(game)])
        if len(moves_scores) == 0:
            return (-1, -1)
        return max(moves_scores, key= moves_scores.get)

    def timeout_test(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def moves(self, game):
        self.timeout_test()
        if self is game.active_player:
            return game.get_legal_moves(self)
        else:
            return game.get_legal_moves(game.get_opponent(self))

    def min_value(self, game, depth):
        self.timeout_test()
        if self.cutoff_test(game, depth):
            return self.evaluate(game)

        value = float("inf")
        for move in self.moves(game):
            self.timeout_test()
            value = min(value, self.max_value(game.forecast_move(move), depth - 1))
        return value

    def max_value(self, game, depth):
        self.timeout_test()
        if self.cutoff_test(game, depth):
            return self.evaluate(game)

        value = float("-inf")
        for move in self.moves(game):
            self.timeout_test()
            value = max(value, self.min_value(game.forecast_move(move), depth - 1))
        return value

    def cutoff_test(self, game, depth):
        self.timeout_test()
        if depth == 0:
            return True
        if game.is_winner(self):
            return True
        if game.is_loser(self):
            return True
        return False

    def evaluate(self, game):
        self.timeout_test()
        return self.score(game, self)

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        try:
            depth = 0
            scores = dict()
            for x in iter(int, 1):
                depth = depth + 1
                best_move = self.alphabeta(game, depth, scores)

        except SearchTimeout:
            # Handle any actions required after timeout as needed
            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, scores, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        self.timeout_test()

        value = float("-inf"); a = float("-inf"); b = float("inf"); max_move = (-1, -1)
        for move in self.moves(game):
            self.timeout_test()
            result = self.min_value(game.forecast_move(move), depth - 1, a, b, scores)
            if result > value:
                value = result; max_move = move
            if value >= b:
                return move
            a = max(value, a)
        return max_move

    def timeout_test(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def moves(self, game):
        self.timeout_test()
        if self is game.active_player:
            return game.get_legal_moves(self)
        else:
            return game.get_legal_moves(game.get_opponent(self))

    def max_value(self, game, depth, alpha, beta, scores):
        self.timeout_test()
        if self.cutoff_test(game, depth):
            return self.evaluate(game)

        value = float("-inf"); a = alpha; b = beta;
        sorted_m = self.sort_moves(game, scores, self.moves(game))
        self.timeout_test()
        for move in sorted_m:
            self.timeout_test()
            new_state = game.forecast_move(move)
            self.timeout_test()
            #new_state.print_board()
            new_value = self.min_value(new_state, depth - 1, a, b, scores)
            scores[new_state.hash()] = new_value
            value = max(value, new_value)
            if value >= b:
                return value
            a = max(value, a)
        return value

    def sort_moves(self, game, scores, moves):
        return moves
        if len(moves) < 1:
            return moves
        valued = dict()
        second_order = []
        for move in moves:
            new_state = game.forecast_move(move)
            new_state_h = new_state.hash()
            if new_state_h in scores:
                #print("Saved")
                valued[move] = scores[new_state_h]
            else:
                second_order.append(move)
        return sorted(valued, key=valued.__getitem__, reverse=True) + second_order



    def min_value(self, game, depth, alpha, beta, scores):
        self.timeout_test()
        if self.cutoff_test(game, depth):
            return self.evaluate(game)

        value = float("inf"); a = alpha; b = beta;
        sorted_m = self.sort_moves(game, scores, self.moves(game))
        self.timeout_test()
        for move in sorted_m:
            self.timeout_test()
            new_state = game.forecast_move(move)
            self.timeout_test()
            new_value = self.max_value(new_state, depth - 1, a, b, scores)
            scores[new_state.hash()] = new_value
            value = min(value, new_value)
            if value <= a:
                return value
            b = min(value, b)
        return value

    def cutoff_test(self, game, depth):
        self.timeout_test()
        if depth == 0:
            return True
        if game.is_winner(self):
            return True
        if game.is_loser(self):
            return True
        return False

    def evaluate(self, game):
        self.timeout_test()
        return self.score(game, self)


class MonteCarloPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    # NO MORE THAN 10 S
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        super(MonteCarloPlayer, self).__init__(search_depth, score_fn, timeout)
        # read file with data to variable
        self.probs = []
        # with open('data.json', 'r') as fp:
        #     self.probs = json.load(fp)


    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        self.setup_probs(game)
        best_move = (-1, -1)
        try:
            best_move = self.return_best(game)
            self.monte_carlo(game)
            print("finish one step")
            print(best_move)

        except SearchTimeout:
            # Handle any actions required after timeout as needed
            if best_move == (-1, -1):
                print("Not solved")
            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def return_best(self, game):
        next_move = self.best_move(game, self.moves(game))
        return next_move

    def monte_carlo(self, game):
        self.timeout_test()
        path = []
        unexplored_state = self.selection(game, path)
        # value should be 1 or 0
        value = self.run_simulation(unexplored_state)
        self.back_propagate(value, path, unexplored_state)

    def selection(self, game, path):
        self.timeout_test()
        if not self.has_state(game):
            return game
        max_value = -1; max_state = None
        path.append(game)
        children = self.moves(game)
        self.timeout_test()
        values = {}
        # return undiscovered state
        for move in children:
            self.timeout_test()
            state = game.forecast_move(move)
            if not self.has_state(state):
                return state
            else:
                new_value = self.get_value_in_state(state)
                values[state] = new_value
        # continue to search using best node.
        if len(values) > 0:
            if self is game.active_player:
                max_state = sorted(values, key=values.__getitem__, reverse=False)[0]
            else:
                max_state = sorted(values, key=values.__getitem__, reverse=True)[0]
        else:
            raise SearchTimeout()
        return self.selection(max_state, path)

    def best_move(self, game, moves):
        self.timeout_test()
        values = {}
        for move in moves:
            #self.timeout_test()
            state = game.forecast_move(move)
            if state.is_winner(self):
                return move
            if self.has_state(state):
                new_value = self.get_value_in_state(state)
                values[move] = new_value
        if len(values) > 0:
            if self is game.active_player:
                max_move = sorted(values, key=values.__getitem__, reverse=False)[0]
            else:
                max_move = sorted(values, key=values.__getitem__, reverse=True)[0]
        else:
            print("problem with values")
            raise SearchTimeout()
        return max_move

    def run_simulation(self, game):
        self.timeout_test()
        if game.is_winner(self):
            return 1
        if game.is_loser(self):
            return 0
        next_states = [];  weights = []
        for move in self.moves(game):
            self.timeout_test()
            state = game.forecast_move(move)
            weight = self.heuristic(state, self)
            self.timeout_test()
            next_states.append(state)
            weights.append(weight)

        p = self.weights_to_probabilities(weights)
        next_state = choice(next_states, p=p)
        return self.run_simulation(next_state)

    def weights_to_probabilities(self, weights):
        self.timeout_test()
        s = sum(weights)
        if s == 0:
            #print("ups")
            return [float(1.0) / len(weights) for x in weights]
        return [float(x) / s for x in weights]

    def heuristic(self, game, player):
        self.timeout_test()
        own_moves = len(game.get_legal_moves(player))
        return float(own_moves)

    def back_propagate(self, value, path, unexplored_state):
        self.timeout_test()
        # is it win or lose?
        self.probs[unexplored_state.move_count][self.state_hash(unexplored_state)] = (value, 1)
        v = value
        for node in path:
            self.timeout_test()
            v = 1 - v
            node_hash = self.state_hash(node)
            win, total = self.probs[node.move_count][node_hash]
            self.probs[node.move_count][node_hash] = (win + v, total + 1)

    def get_value_in_state(self, state):
        win, total = self.probs[state.move_count][self.state_hash(state.hash)]
        n = sum([x[1] for x in self.probs[state.move_count].values()])
        return (win / float(total) + sqrt(2 * log(total) / n))

    def has_state(self, state):
        return (self.state_hash(state) in self.probs[state.move_count])

    def setup_probs(self, game):
        self.timeout_test()
        if len(self.probs) > 0:
            return
        for i in range(game.width * game.height):
            self.probs.append(dict())
        self.probs[game.move_count][self.state_hash(game)] = (0, 0)

    def state_hash(self, state):
        return state.hash()

    def timeout_test(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def moves(self, game):
        self.timeout_test()
        if self is game.active_player:
            return game.get_legal_moves(self)
        else:
            return game.get_legal_moves(game.get_opponent(self))

    # think about saving probs by move_count (layers)
    # saves probabilities to file and run it during simulation
    def save(self):
        with open('data.json', 'w') as fp:
            json.dump(self.probs, fp)


