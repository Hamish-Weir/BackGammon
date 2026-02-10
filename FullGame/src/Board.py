from copy import deepcopy
import numpy as np

BOARD_SIZE = 24
TOTAL_PLAYER_PIECES = 15
HOME_SIZE = 6


BOARD_START = 0             # 0
BOARD_END   = BOARD_SIZE-1  # 23
BAR_START   = BOARD_END+1   # 24
BAR_END     = BOARD_END+2   # 25
OFF_START   = BOARD_END+3   # 26
OFF_END     = BOARD_END+4   # 27

P1LOGICALBAR = -1
P2LOGICALBAR = 24

P1LOGICALOFF = 24
P2LOGICALOFF = -1

P1BAR = BAR_START    # 24
P2BAR = BAR_END      # 25

P1OFF = OFF_START    # 26
P2OFF = OFF_END      # 27



P1OUT_START = BOARD_START                   # 0
P1OUT_END   = BOARD_END - HOME_SIZE         # 17
P1HOME_START = BOARD_END - HOME_SIZE + 1    # 18
P1HOME_END   = BOARD_END                    # 23

P2OUT_START = BOARD_END                     # 23             
P2OUT_END   = BOARD_START + HOME_SIZE       # 6
P2HOME_START = BOARD_START + HOME_SIZE - 1  # 5
P2HOME_END   = BOARD_START                  # 0

class Board():

    # Positive = RED
    # Negative = BLUE
    def __init__(self):
        self._tiles = np.array([
            2,  0,  0,  0,  0, -5,  # Red Home
            0, -3,  0,  0,  0,  5,
           -5,  0,  0,  0,  3,  0,
            5,  0,  0,  0,  0, -2,  # Blue Home
            0,  0,                  # Red/Blue Bar
            0,  0])                 # Red/Blue Off
        
        assert sum(self._tiles[self._tiles>0]) == TOTAL_PLAYER_PIECES, f"RED must have ({TOTAL_PLAYER_PIECES}) pieces ({sum(self._tiles[self._tiles>0])})"
        assert sum(self._tiles[self._tiles<0]) == -TOTAL_PLAYER_PIECES, f"BLUE must have ({TOTAL_PLAYER_PIECES}) pieces ({sum(self._tiles[self._tiles<0])})"
        
    def set(self,arr):
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 28, "Array must be of length 28"
        assert all(isinstance(x, int) for x in arr), "All elements must be integers"
        assert sum(arr>0) == TOTAL_PLAYER_PIECES, f"RED must have ({TOTAL_PLAYER_PIECES}) pieces"
        assert sum(arr<0) == TOTAL_PLAYER_PIECES, f"BLUE must have ({TOTAL_PLAYER_PIECES}) pieces"
        
        self._tiles = arr.copy()
    
    def get(self):
        return self._tiles.copy()

    def distance(start, end, player):
        assert player == 1 or player == -1, "Player must be 1 or -1"
        assert (start <= BOARD_END and start >= BOARD_START) or (start <= BAR_END and start >= BAR_START), f"Start ({start}) in must be in Valid Range"
        assert (end   <= BOARD_END and end   >= BOARD_START) or (end   <= OFF_END and end   >= OFF_START), f"End ({end}) in must be in Valid Range"
        assert not (start == P1BAR and start == P2OFF), f"Invalid Combination of start ({start}) and end ({end})"
        assert not (start == P2BAR and start == P1OFF), f"Invalid Combination of start ({start}) and end ({end})"

        if player == 1:
            match (start,end):
                case (P1BAR,  P1OFF ):  # Whole Board Jump
                    return BOARD_SIZE+1
                case (P1BAR,  _           ):  # Logical Start = -1 
                    return end+1
                case (_,            P1OFF ):  # Logical End = 24
                    return BOARD_SIZE-start
                case _:                             # Normal Move
                    return end-start
        else:
            match (start,end):
                case (P2BAR,  P2OFF ):  # Whole Board Jump
                    return BOARD_SIZE+1
                case (P2BAR,  _           ):  # Logical Start = -1 
                    return BOARD_SIZE-end
                case (_,            P2OFF ):  # Logical End = 24
                    return start+1
                case _:                             # Normal Move
                    return start-end

    def end_point(start,die,player):
        assert player == 1 or player == -1, f"Player ({player}) must be 1 or -1"
        assert (start <= BOARD_END and start >= BOARD_START) or (start <= BAR_END and start >= BAR_START), f"Start ({start}) in must be in Valid Range"
        assert (die <= 6 and die >= 1), f"die ({die}) must be in 1-6"

        if player == 1:
            if start == P1BAR:            # From Bar
                return die-1
            elif start + die > BOARD_END:    # To Off
                return P1OFF
            else:                               # Normal
                return start+die
        else:
            if start == P2BAR:    # From Bar
                return BOARD_SIZE-die
            elif start - die < 0:       # To Off
                return P2OFF
            else:                       # Normal
                return start-die

    def get_legal_moves(self,die,player):

        def has_my_piece(pos,player):
            if player == 1:
                if self._tiles[pos] > 0:
                    return True
                return False
            else:
                if self._tiles[pos] < 0:
                    return True
                return False

        def can_land(pos,player):
            if player == 1:
                if self._tiles[pos] > -2:
                    return True
                return False
            else:
                if self._tiles[pos] < 2:
                    return True
                return False

        def player_bar(player):
            if player == 1:
                return P1BAR
            return P2BAR

        def player_off(player):
            if player == 1:
                return P1OFF
            return P2OFF

        def can_bear_off(player):
            if player == 1:
                return sum(self._tiles[P1OUT_START:P1OUT_END+1] > 0) == 0
            return sum(self._tiles[P2OUT_END:P2OUT_START+1] < 0) == 0

        def direct_move_off_point(die,player):
            if player == 1:
                return (P1LOGICALOFF - die)
            return (die - P2LOGICALOFF)
        
        def all_start_pips(player):
            if player == 1:
                return np.where(self._tiles[0:24]>0)[0]
            return np.where(self._tiles[0:24]<0)[0]

        def get_mask(s,d,player):
            
            if player == 1:
                e = s+d
                return (e >= BOARD_START) & (e <= BOARD_END) & ((self._tiles[e.clip(BOARD_START,BOARD_END)]) > -2)
            e = s-d
            return (e >= BOARD_START) & (e <= BOARD_END) & ((self._tiles[e.clip(BOARD_START,BOARD_END)]) < 2)

        assert player == 1 or player == -1, f"Player ({player}) must be 1 or -1"
        assert (die <= 6 and die >= 1), f"die ({die}) must be in 1-6"
        assert sum(self._tiles[self._tiles>0]) == TOTAL_PLAYER_PIECES, f"RED must have ({TOTAL_PLAYER_PIECES}) pieces ({sum(self._tiles[self._tiles>0])})"
        assert sum(self._tiles[self._tiles<0]) == -TOTAL_PLAYER_PIECES, f"BLUE must have ({TOTAL_PLAYER_PIECES}) pieces ({sum(self._tiles[self._tiles<0])})"
        
        moveSet = set()

        if has_my_piece(player_bar(player),player): # Piece on Bar, Must Move First
            end_pip = Board.end_point(P1BAR,die,player)
            if can_land(end_pip,player):
                moveSet.add((player_bar(player),die))
        else:
            if can_bear_off(player):
                start_pip = direct_move_off_point(die,player)
                if has_my_piece(start_pip,player):
                    moveSet.add((start_pip,die))
            possible_start_pips = all_start_pips(player)

            start_pips = possible_start_pips
            end_pips = start_pips - die

            mask = get_mask(start_pips,die,player)

            masked_start_pips = start_pips[mask]

            new_moves = [(int(start_pip),int(die)) for start_pip in masked_start_pips]
            moveSet.update(new_moves)

        return sorted(list(moveSet))
    
    def get_legal_movesequences(self,die1,die2,player):

        moveSequenceSet = set()
        moveSequenceList = []

        if die1 == die2:
            M1 = self.get_legal_moves(die1,player)
            if M1:
                B1 = Board()
                B1_A = self.get()
                for m1 in M1:
                    B1.set(B1_A)
                    B1.do_move(m1,player)
                    M2 = B1.get_legal_moves(die1,player)
                    if M2:
                        B2 = Board()
                        B2_A = B1.get()
                        for m2 in M2:
                            B2.set(B2_A)
                            B2.do_move(m2,player)
                            M3 = B2.get_legal_moves(die1,player)
                            if M3:
                                B3 = Board()
                                B3_A = B2.get()
                                for m3 in M3:
                                    B3.set(B3_A)
                                    B3.do_move(m3,player)
                                    M4 = B3.get_legal_moves(die1,player)
                                    if M4:
                                        for m4 in M4:
                                            ms = sorted([m1,m2,m3,m4])
                                            if not(ms in moveSequenceSet):
                                                moveSequenceSet.add(tuple(ms))
                                                moveSequenceList.append(ms)
                                    else:
                                        ms = sorted([m1,m2,m3])
                                        if not(ms in moveSequenceSet):
                                            moveSequenceSet.add(tuple(ms))
                                            moveSequenceList.append(ms)
                            else:
                                ms = sorted([m1,m2])
                                if not(ms in moveSequenceSet):
                                    moveSequenceSet.add(tuple(ms))
                                    moveSequenceList.append(ms)
                    else:
                        ms = sorted([m1])
                        if not(ms in moveSequenceSet):
                            moveSequenceSet.add(tuple(ms))
                            moveSequenceList.append(ms)
            else:
                ms = sorted([])
                if not(ms in moveSequenceSet):
                    moveSequenceSet.add(tuple(ms))
                    moveSequenceList.append(ms)

    def do_move(self,move,player):
        if move:
            start_pip, die = move
            end_pip = Board.end_point(start_pip,die,player)

            if player == 1:
                # if killable move dead piece to BAR
                if self._tiles[end_pip]==-1:
                    self._tiles[end_pip] = 0
                    self._tiles[Board.P2BAR] = self._tiles[Board.P2BAR]-1

                self._tiles[start_pip] = self._tiles[start_pip]-1
                self._tiles[end_pip] = self._tiles[end_pip]+1
            else:
                # if killable move dead piece to BAR
                if self._tiles[end_pip]== 1:
                    self._tiles[end_pip] = 0
                    self._tiles[Board.P1BAR] = self._tiles[Board.P1BAR]+1

                self._tiles[start_pip] = self._tiles[start_pip]+1
                self._tiles[end_pip] = self._tiles[end_pip]-1

    def do_move_sequence(self,move_sequence,player):
        for move in move_sequence:
            self.do_move(move,player)

    def __str__(self) -> str:
        RED = "\033[91m"
        BLUE = "\033[0;34m"
        YELLOW = "\033[1;33m"
        END = "\033[0m"

        BOARD_COLOUR = YELLOW
        
        def get_spaced_str(list: list[int]):
            return " ".join([f"{n:4d}" for n in list])

        def get_spaced_str_board(list: list[int]):
            list2 = []
            for n in list:
                if n == 0:
                    list2.append(f"    ")
                elif n < 0:
                    list2.append(f"{BLUE}{-n:4d}{END}")
                else:
                    list2.append(f"{RED}{n:4d}{END}")
            return " ".join(list2)

        board_str = f"""                {BLUE}BLUE Home                          {BLUE}Out
        {BOARD_COLOUR}OFF     {get_spaced_str([1, 2, 3, 4, 5, 6])}      {get_spaced_str([7, 8, 9, 10, 11, 12])}     BAR
                {BOARD_COLOUR}+----+----+----+----+----+----+    +----+----+----+----+----+----+
        {BLUE}{-self._tiles[P2OFF]:4d}    {get_spaced_str_board(self._tiles[0:6])}      {get_spaced_str_board(self._tiles[6:12])}   {BLUE}{-self._tiles[P2BAR]:4d}
                {BOARD_COLOUR}+----+----+----+----+----+----+    +----+----+----+----+----+----+
        {RED}{self._tiles[P1OFF]:4d}    {get_spaced_str_board(self._tiles[23:17:-1])}      {get_spaced_str_board(self._tiles[17:11:-1])}   {RED}{self._tiles[P1BAR]:4d}
                {BOARD_COLOUR}+----+----+----+----+----+----+    +----+----+----+----+----+----+
                {get_spaced_str([24, 23, 22, 21, 20, 19])}      {get_spaced_str([18, 17, 16, 15, 14, 13])}
                {RED}Red Home                           {RED}Out{END}
        """
        
        return board_str

    def get_tiles(self):
        return self._tiles.copy()