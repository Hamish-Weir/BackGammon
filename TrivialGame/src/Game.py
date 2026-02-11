from copy import deepcopy
import logging
import os
from random import randint
import sys
from time import perf_counter_ns as time
from typing import TextIO

from AgentBase import AgentBase
from Player import Player
from Board import Board, P1BAR,P2BAR,P1OFF,P2OFF
import random

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(levelname)s]-%(asctime)s - %(message)s", level=logging.INFO
)

class Game:
    _turn:int

    def __init__(
        self,
        player1: Player,
        player2: Player,
        silent: bool = False,
        timelimitOff = False
    ):
        self._turn = 0
        self._start_time = time()
        self.random = random

        self._board = Board()

        self.current_player = 1
        self.player1 = player1
        self.player2 = player2
        
        self.players = {
            1: self.player1,
            -1: self.player2,
        }

        
        self.timelimitOff = timelimitOff
        logDest = sys.stderr
        if silent:
            logger.setLevel(logging.CRITICAL)
            logDest = os.devnull

        if logDest != sys.stderr:
            self.logDest = open(logDest, "w")
        else:
            self.logDest = logDest

    @property
    def turn(self):
        return self._turn

    @property
    def board(self):
        return self._board

    def ms_to_str(ms,player):
        ms_str = "["
        for m in ms:
            if m[0] == P1BAR or m[0] == P2BAR:
                s = "BAR"
            else:
                s = f"{m[0]+1:3d}"

            if Board.end_point(m[0],m[1],player) == P1OFF or Board.end_point(m[0],m[1],player) == P2OFF:
                e = "OFF"
            else:
                e = f"{Board.end_point(m[0],m[1],player)+1:3d}"

            m_str = f"({s}, {e}, {m[1]}), "
            ms_str += m_str
        ms_str += "]"    
        return ms_str  


    def run(self):
        assert issubclass(type(self.players[1].agent), AgentBase)
        assert issubclass(type(self.players[-1].agent), AgentBase)

        logger.info("Game started")
        self._play()

        if self.logDest != sys.stderr:
            self.logDest.close()

        print(self.board)

        return self.board.get_winner()
    
    def _play(self):

        opponentMove = None
        
        while True:
            self._turn += 1
            die1, die2 = self.random.randint(1,6),self.random.randint(1,6)
            currentPlayer: Player = self.players[self.current_player]
            playerAgent = currentPlayer.agent
            logger.info(f"Turn {self.turn}: player {currentPlayer.name}")
            logger.info(f"Starting Board:\n{str(self.board)}")
            logger.info(f"Dice Roll: {die1}, {die2}")
            
            playerBoard = Board()
            playerBoard.set(self.board.get())

            start = time()
            ms = playerAgent.make_move(playerBoard, die1, die2, opponentMove)
            end = time()

            currentPlayer.move_time += end - start

            logger.debug(
                f"Player {currentPlayer.name}; Move time: {Game.ns_to_s(currentPlayer.move_time)}s"
            )
            logger.info(f"Player {currentPlayer.name}; Move Sequence: {self.current_player}\n{Game.ms_to_str(ms,self.current_player)}")
            if not self.timelimitOff:
                if currentPlayer.move_time > Game.MAXIMUM_TIME:
                    logger.info(f"Player {currentPlayer.name} timed out")
                    raise Exception("Player Timed Out")
            
            if self.is_valid_move_sequence(die1,die2,ms):
                logger.debug("Move is valid")
                self._make_move(ms)
                opponentMove = ms
            else:
                logger.info(f"Player {currentPlayer.name} made an illegal move")
                raise Exception("Illegal Move")
            if self.board.get_winner() != 0:
                break

            
            logger.info("")
            self.current_player = -self.current_player

        return self._end_game()

    def _end_game(self):

        total_time = time() - self._start_time

        logger.info("Game over")
        logger.info(f"Final Board:\n{str(self.board)}")
        logger.info(f"Total time: {Game.ns_to_s(total_time)}s")

        logger.info(f"Player {self.players[self.current_player].name} has won")
        winner = self.players[self.current_player].name

        for p in self.players.values():
            print(f"{p.name},{p.move_time}", file=self.logDest)
        print(f"winner,{winner}", file=self.logDest)
        logger.info(f"Total Game Time: {Game.ns_to_s(total_time)}s")

    def is_valid_move_sequence(self,die1, die2, ms):
        return True

    def _make_move(self, ms):
        """Performs a valid move on the board, then prints its results."""

        logger.debug(f"Move made: {ms}")
        current_player = self.players[self.current_player]

        self.board.do_move_sequence(ms,self.current_player)

        print(current_player)
        print(
            f"{self.turn},{current_player.name},{self.current_player},{ms},{current_player.move_time}",
            file=self.logDest,
        )

  

    def ns_to_s(t):
        """Method for standardised nanosecond to second conversion."""
        return int(t / 10**6) / 10**3