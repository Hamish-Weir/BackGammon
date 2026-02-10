from Board import Board
import pytest

def test_init():
    b = Board()
    c = [   2,  0,  0,  0,  0, -5,  # Red Home
            0, -3,  0,  0,  0,  5,
           -5,  0,  0,  0,  3,  0,
            5,  0,  0,  0,  0, -2,  # Blue Home
            0,  0,                  # Red/Blue Bar
            0,  0]

    assert len(b.tiles()) == len(c)
    assert all(x == y for x, y in zip(b.tiles(), c))
    
def test_distance_player():
    with pytest.raises(AssertionError):
        Board.distance(0,1,0)
    assert Board.distance(0,1,1) == 1
    assert Board.distance(1,0,-1) == 1

def test_distance_normal():
    for i in range(0,24):
        for j in range(0,24):
            assert Board.distance(i,j,1) == j-i
    
    for i in range(0,24):
        for j in range(0,24):
            assert Board.distance(i,j,1) == j-i

def test_distance_oob():
    with pytest.raises(AssertionError):
        Board.distance(-1,1,1)
    with pytest.raises(AssertionError):
        Board.distance(-1,1,-1) 
    with pytest.raises(AssertionError):
        Board.distance(1,-1,1)
    with pytest.raises(AssertionError):
        Board.distance(1,-1,-1) 
    with pytest.raises(AssertionError):
        Board.distance(-1,-1,1)
    with pytest.raises(AssertionError):
        Board.distance(-1,-1,-1) 

def test_distance_off():
    for i in range(0,24):
        assert Board.distance(i,Board.P1OFF,1) == 24-i
    for i in range(0,24):
        assert Board.distance(i,Board.P2OFF,-1) == i+1
    
def test_distance_bar():
    for j in range(0,24):
        assert Board.distance(Board.P1BAR,j,1) == j+1
    for j in range(0,24):
        assert Board.distance(Board.P2BAR,j,-1) == 24-j

def test_distance_full_board():
    assert Board.distance(Board.P1BAR,Board.P1OFF,1) == 25
    assert Board.distance(Board.P2BAR,Board.P2OFF,-1) == 25

