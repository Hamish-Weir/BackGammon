from copy import deepcopy
import numpy as np

BOARD_SIZE = 6
TOTAL_PLAYER_PIECES = 2
HOME_SIZE = 2
DIE_SIZE = 2

BOARD_START = 2            
BOARD_END   = BOARD_START + BOARD_SIZE-1  

GAME_SIZE = BOARD_SIZE + 4

P2OFF = BOARD_START-2   
P1BAR = BOARD_START-1  

P2BAR = BOARD_END+1    
P1OFF = BOARD_END+2   




P1OUT_START = BOARD_START                   
P1OUT_END   = BOARD_END - HOME_SIZE         
P1HOME_START = BOARD_END - HOME_SIZE + 1    
P1HOME_END   = BOARD_END                    

P2OUT_START = BOARD_END                              
P2OUT_END   = BOARD_START + HOME_SIZE    
P2HOME_START = BOARD_START + HOME_SIZE - 1 
P2HOME_END   = BOARD_START                

P1LOGICALBAR = BOARD_START-1
P2LOGICALBAR = BOARD_END+1

P1LOGICALOFF = BOARD_END+1
P2LOGICALOFF = BOARD_START-1

class Board():

    # Positive = RED
    # Negative = BLUE
    def __init__(self):
        self._tiles = np.array([
            0,  0,  # Blue Off, Red Bar
            2,  0,  # Red Home
            0,  0,
            0, -2,  # Blue Home
            0,  0   # Blue Bar, Red Off
            ],dtype=np.int8)                 
        
        assert sum(self._tiles[self._tiles>0]) == TOTAL_PLAYER_PIECES, f"RED must have ({TOTAL_PLAYER_PIECES}) pieces ({sum(self._tiles[self._tiles>0])})"
        assert sum(self._tiles[self._tiles<0]) == -TOTAL_PLAYER_PIECES, f"BLUE must have ({TOTAL_PLAYER_PIECES}) pieces ({sum(self._tiles[self._tiles<0])})"
        
    def set(self,arr):
        assert isinstance(arr, np.ndarray)
        assert len(arr) == BOARD_SIZE+4, f"Array length ({len(arr)}) must be {BOARD_SIZE+4}"
        # assert all(isinstance(x, int) or isinstance(x, np.int_) for x in arr), f"All elements must be integers {type(arr[0])}"
        assert sum(arr[arr>0]) == TOTAL_PLAYER_PIECES, f"RED must have ({TOTAL_PLAYER_PIECES}) pieces ({sum(arr[arr>0])})"
        assert sum(arr[arr<0]) == -TOTAL_PLAYER_PIECES, f"BLUE must have ({TOTAL_PLAYER_PIECES}) pieces({sum(arr[arr>0])})"
        
        self._tiles = np.array(arr,dtype=np.int8)
    
    def get(self):
        return self._tiles.copy()

    def distance(start, end, player):
        assert player == 1 or player == -1, "Player must be 1 or -1"
        assert (start <= BOARD_END and start >= BOARD_START) or (start == P1BAR or start == P2BAR), f"Start ({start}) in must be in Valid Range"
        assert (end   <= BOARD_END and end   >= BOARD_START) or (end   == P1OFF or end   == P2OFF), f"End ({end}) in must be in Valid Range"
        assert not (start == P1BAR and start == P2OFF), f"Invalid Combination of start ({start}) and end ({end})"
        assert not (start == P2BAR and start == P1OFF), f"Invalid Combination of start ({start}) and end ({end})"

        if player == 1:
            if start == P1BAR and end == P1OFF:
                return BOARD_SIZE + 1
            elif start == P1BAR:
                return end - P1LOGICALBAR
            elif end == P1OFF:
                return P1LOGICALOFF - start
            else:
                return end - start
        else:
            if start == P2BAR and end == P2OFF:  # Whole Board Jump
                return BOARD_SIZE+1
            elif start == P2BAR:  # Logical Start = -1 
                return P2LOGICALBAR-end
            elif end == P2OFF:  # Logical End = 24
                return start - P2LOGICALOFF
            else:                             # Normal Move
                return start-end

    def end_point(start,die,player):
        assert player == 1 or player == -1, f"Player ({player}) must be 1 or -1"
        assert (start <= BOARD_END and start >= BOARD_START) or (start == P1BAR or start == P2BAR), f"Start ({start}) in must be in Valid Range"
        assert (die <= DIE_SIZE and die >= 1), f"die ({die}) must be in 1-{DIE_SIZE}"

        if player == 1:
            if start == P1BAR:            # From Bar
                return die + P1LOGICALBAR
            elif start + die > BOARD_END:    # To Off
                return P1OFF
            else:                               # Normal
                return start+die
        else:
            if start == P2BAR:    # From Bar
                return P2LOGICALBAR - die
            elif start - die < BOARD_START:       # To Off
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
            return (die + P2LOGICALOFF)
        
        def all_start_pips(player):
            if player == 1:
                return BOARD_START+np.where(self._tiles[BOARD_START:BOARD_END+1]>0)[0]
            return BOARD_START+np.where(self._tiles[BOARD_START:BOARD_END+1]<0)[0]

        def get_mask(s,d,player):
            
            if player == 1:
                e = s+d
                return (e >= BOARD_START) & (e <= BOARD_END) & ((self._tiles[e.clip(BOARD_START,BOARD_END)]) > -2)
            e = s-d
            return (e >= BOARD_START) & (e <= BOARD_END) & ((self._tiles[e.clip(BOARD_START,BOARD_END)]) < 2)

        def get_furthest(player):
            if player==1:
                pips_in_home = np.where(self._tiles[P1HOME_START:P1HOME_END+1]>0)[0]
                if not len(pips_in_home) == 0:
                    return P1HOME_START+np.min(pips_in_home)
                else:
                    raise Exception(f"No pips in home, this should not happen \n{self}")
            pips_in_home = np.where(self._tiles[P2HOME_END:P2HOME_START+1]<0)[0]
            if not len(pips_in_home) == 0:
                return P2HOME_END+np.max(pips_in_home)
            else:    
                raise Exception(f"No pips in home, this should not happen \n{self}")


        assert player == 1 or player == -1, f"Player ({player}) must be 1 or -1"
        assert (die <= DIE_SIZE and die >= 1), f"die ({die}) must be in 1-{DIE_SIZE}"
        assert sum(self._tiles[self._tiles>0]) == TOTAL_PLAYER_PIECES, f"RED must have ({TOTAL_PLAYER_PIECES}) pieces ({sum(self._tiles[self._tiles>0])}) {self}"
        assert sum(self._tiles[self._tiles<0]) == -TOTAL_PLAYER_PIECES, f"BLUE must have ({TOTAL_PLAYER_PIECES}) pieces ({sum(self._tiles[self._tiles<0])})"
        
        if self.get_winner() != 0:
            return []

        moveSet = set()

        if has_my_piece(player_bar(player),player): # Piece on Bar, Must Move First
            end_pip = Board.end_point(player_bar(player),die,player)
            if can_land(end_pip,player):
                moveSet.add((player_bar(player),die))
        else:
            if can_bear_off(player):
                start_pip = direct_move_off_point(die,player)
                if has_my_piece(start_pip,player):
                    moveSet.add((start_pip,die))
                else:
                    
                    start_pip = get_furthest(player) #
                    if Board.distance(start_pip,player_off(player),player) < die:
                        moveSet.add((start_pip,die))

            possible_start_pips = all_start_pips(player)

            start_pips = possible_start_pips

            mask = get_mask(start_pips,die,player)

            masked_start_pips = start_pips[mask]

            new_moves = [(int(s),int(die)) for s in masked_start_pips]
            moveSet.update(new_moves)

        legal_moves = sorted(list(moveSet))
        return legal_moves
    
    def get_legal_movesequences(self,player):

        reversed_order = False if player == 1 else True

        moveSequenceSet = set()
        moveSequenceList = []

        B1 = Board()

        for die1 in range(1,DIE_SIZE+1):
            for die2 in range(die1,DIE_SIZE+1):
                if die1 == die2:
                    M1 = self.get_legal_moves(die1,player)
                    if M1:
                        B1_A = self.get()
                        for m1 in M1:
                            B1.set(B1_A)
                            B1.do_move(m1,player)
                            M2 = B1.get_legal_moves(die1,player)
                            if M2:
                                for m2 in M2:
                                    ms = [m1,m2]
                                    ms = sorted(ms,reverse=reversed_order)
                                    tms = tuple(ms)
                                    if not(tms in moveSequenceSet):
                                        moveSequenceSet.add(tms)
                                        moveSequenceList.append(ms)
                            else:
                                ms = [m1]
                                ms = sorted(ms,reverse=reversed_order)
                                tms = tuple(ms)
                                if not(tms in moveSequenceSet):
                                    moveSequenceSet.add(tms)
                                    moveSequenceList.append(ms)

                else: # Not Double
                    M1 = self.get_legal_moves(die1,player)
                    if M1:
                        B1_A = self.get()
                        for m1 in M1:
                            B1.set(B1_A)
                            B1.do_move(m1,player)
                            M2 = B1.get_legal_moves(die2,player)
                            if M2:
                                for m2 in M2:
                                    ms = [m1,m2]
                                    ms = sorted(ms,reverse=reversed_order)
                                    tms = tuple(ms)
                                    if not(tms in moveSequenceSet):
                                        moveSequenceSet.add(tms)
                                        moveSequenceList.append(ms)
                            else:
                                ms = [m1]
                                ms = sorted(ms,reverse=reversed_order)
                                tms = tuple(ms)
                                if not(tms in moveSequenceSet):
                                    moveSequenceSet.add(tms)
                                    moveSequenceList.append(ms)

                    M1 = self.get_legal_moves(die2,player)
                    if M1:
                        B1_A = self.get()
                        for m1 in M1:
                            B1.set(B1_A)
                            B1.do_move(m1,player)
                            M2 = B1.get_legal_moves(die1,player)
                            if M2:
                                for m2 in M2:
                                    ms = [m1,m2]
                                    ms = sorted(ms,reverse=reversed_order)
                                    tms = tuple(ms)
                                    if not(tms in moveSequenceSet):
                                        moveSequenceSet.add(tms)
                                        moveSequenceList.append(ms)
                            else:
                                ms = [m1]
                                ms = sorted(ms,reverse=reversed_order)
                                tms = tuple(ms)
                                if not(tms in moveSequenceSet):
                                    moveSequenceSet.add(tms)
                                    moveSequenceList.append(ms)


            if len(moveSequenceSet) == 0:
                moveSequenceSet.add(())
                moveSequenceList.append([])


        
        return moveSequenceList

    def do_move(self,move,player):
        if move:
            start_pip, die = move
            end_pip = Board.end_point(start_pip,die,player)

            if player == 1:
                # if killable move dead piece to BAR
                if self._tiles[end_pip]==-1:
                    self._tiles[end_pip] = 0
                    self._tiles[P2BAR] = self._tiles[P2BAR]-1

                self._tiles[start_pip] = self._tiles[start_pip]-1
                self._tiles[end_pip] = self._tiles[end_pip]+1
            else:
                # if killable move dead piece to BAR
                if self._tiles[end_pip]== 1:
                    self._tiles[end_pip] = 0
                    self._tiles[P1BAR] = self._tiles[P1BAR]+1

                self._tiles[start_pip] = self._tiles[start_pip]+1
                self._tiles[end_pip] = self._tiles[end_pip]-1

    def do_move_sequence(self,move_sequence,player):
        for move in move_sequence:
            self.do_move(move,player)

    def __str__(self) -> str:
        RED = "\033[91m"
        BLU = "\033[0;34m"
        YELLOW = "\033[1;33m"
        END = "\033[0m"

        BOA = YELLOW
        
        def get_spaced_str(list: list[int]):
            return " ".join([f"{n:4d}" for n in list])

        def get_spaced_str_board(list: list[int]):
            list2 = []
            for n in list:
                if n == 0:
                    list2.append(f"    ")
                elif n < 0:
                    list2.append(f"{BLU}{-n:4d}{END}")
                else:
                    list2.append(f"{RED}{n:4d}{END}")
            return " ".join(list2)

        board_str  = f""
        board_str += f"{BLU}BLUE OFF     {BOA}| {BLU}BLUE Home           {BLU}Out         {RED}RED Home{BOA}|      {RED}RED OFF\n"
        board_str += f"     {RED}RED BAR {BOA}| {BOA}{get_spaced_str([1, 2])}      {get_spaced_str([3, 4])}      {get_spaced_str([5, 6])} {BOA}| {BLU}BLUE BAR     \n"
        board_str += f"             {BOA}|+----+----+    +----+----+    +----+----+|\n"
        board_str += f"{BLU}{-self._tiles[P2OFF]:>4d}{RED}{self._tiles[P1BAR]:>4d}     {BOA}|{get_spaced_str_board(self._tiles[BOARD_START:P2HOME_START+1])}      {get_spaced_str_board(self._tiles[P2OUT_END:P1OUT_END+1])}      {get_spaced_str_board(self._tiles[P1HOME_START:P1HOME_END+1])}  {BOA}| {BLU}{-self._tiles[P2BAR]:>4d}{RED}{self._tiles[P1OFF]:>4d}      \n"
        board_str += f"{END}"

        return board_str

    def get_winner(self):
        if self._tiles[P1OFF] == TOTAL_PLAYER_PIECES:
            return 1
        elif self._tiles[P2OFF] == -TOTAL_PLAYER_PIECES:
            return -1
        return 0
    
    def __eq__(self,other):
        if not isinstance(other, Board):
            return 0
        return np.array_equal(self._tiles,other._tiles)

    