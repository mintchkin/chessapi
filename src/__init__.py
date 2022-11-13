from __future__ import annotations
from asyncio import Queue
from dataclasses import dataclass
import re
from typing import ClassVar, Iterable, Type


class Player:
    def __init__(self):
        self.online = True


class Game:
    def __init__(self, player1: Player, player2: Player | None = None):
        self.player1 = player1
        self.player2 = player2

    def awaiting_opponent(self):
        return self.player1.online and not self.started()

    def started(self):
        return self.player2 is not None

    def is_playing(self, player: Player):
        return player == self.player1 or player == self.player2

    def set_opponent(self, player: Player):
        assert not self.started()
        assert not self.is_playing(player)
        self.player2 = player


class Piece:
    is_white: bool

    def opposes(self, other: Piece):
        return self.is_white != other.is_white

    def moves(self, board: Board) -> Iterable[Pos]:
        return []

    def symbol(self) -> str:
        raise NotImplementedError()


class Pawn(Piece):
    def __init__(self, is_white: bool):
        self.is_white = is_white

    def symbol(self):
        return ["♟", "♙"][self.is_white]

    def moves(self, board: Board):
        pos = board.find(self)

        advance = pos.adv(self.is_white)
        if board.has_pos(advance) and board.at(advance) is None:
            yield advance

        attack_qs = advance.qside()
        if (
            board.has_pos(attack_qs)
            and isinstance(piece := board.at(attack_qs), Piece)
            and self.opposes(piece)
        ):
            yield attack_qs

        attack_ks = advance.kside()
        if (
            board.has_pos(attack_ks)
            and isinstance(piece := board.at(attack_ks), Piece)
            and self.opposes(piece)
        ):
            yield attack_ks

        if self.is_white and pos.irow == 2 or not self.is_white and pos.irow == 7:
            yield advance.adv(self.is_white)

        # TODO: En passant


class Knight(Piece):
    def __init__(self, is_white: bool):
        self.is_white = is_white

    def symbol(self):
        return ["♞", "♘"][self.is_white]

    def moves(self, board: Board):
        pos = board.find(self)
        moves = [
            pos.adv(self.is_white).adv(self.is_white).kside(),
            pos.adv(self.is_white).adv(self.is_white).qside(),
            pos.adv(self.is_white).kside().kside(),
            pos.adv(self.is_white).qside().qside(),
            pos.ret(self.is_white).ret(self.is_white).kside(),
            pos.ret(self.is_white).ret(self.is_white).qside(),
            pos.ret(self.is_white).kside().kside(),
            pos.ret(self.is_white).qside().qside(),
        ]
        for move in moves:
            if not board.has_pos(move):
                continue

            if (other := board.at(move)) is None or self.opposes(other):
                yield move


class Bishop(Piece):
    def __init__(self, is_white: bool):
        self.is_white = is_white

    def symbol(self):
        return ["♝", "♗"][self.is_white]

    def moves(self, board: Board):
        pos = board.find(self)

        move = pos
        while board.has_pos(move := move.adv(self.is_white).qside()):
            if (other := board.at(move)) is None or self.opposes(other):
                yield move
            if other is not None:
                break

        move = pos
        while board.has_pos(move := move.adv(self.is_white).kside()):
            if (other := board.at(move)) is None or self.opposes(other):
                yield move
            if other is not None:
                break

        move = pos
        while board.has_pos(move := move.ret(self.is_white).qside()):
            if (other := board.at(move)) is None or self.opposes(other):
                yield move
            if other is not None:
                break

        move = pos
        while board.has_pos(move := move.ret(self.is_white).kside()):
            if (other := board.at(move)) is None or self.opposes(other):
                yield move
            if other is not None:
                break


class Rook(Piece):
    def __init__(self, is_white: bool):
        self.is_white = is_white

    def symbol(self):
        return ["♜", "♖"][self.is_white]

    def moves(self, board: Board):
        pos = board.find(self)

        move = pos
        while board.has_pos(move := move.adv(self.is_white)):
            if (other := board.at(move)) is None or self.opposes(other):
                yield move
            if other is not None:
                break

        move = pos
        while board.has_pos(move := move.ret(self.is_white)):
            if (other := board.at(move)) is None or self.opposes(other):
                yield move
            if other is not None:
                break

        move = pos
        while board.has_pos(move := move.qside()):
            if (other := board.at(move)) is None or self.opposes(other):
                yield move
            if other is not None:
                break

        move = pos
        while board.has_pos(move := move.kside()):
            if (other := board.at(move)) is None or self.opposes(other):
                yield move
            if other is not None:
                break


class Queen(Piece):
    def __init__(self, is_white: bool):
        self.is_white = is_white

    def symbol(self):
        return ["♛", "♕"][self.is_white]

    def moves(self, board: Board):
        pos = board.find(self)

        move = pos
        while board.has_pos(move := move.adv(self.is_white).qside()):
            if (other := board.at(move)) is None or self.opposes(other):
                yield move
            if other is not None:
                break

        move = pos
        while board.has_pos(move := move.adv(self.is_white).kside()):
            if (other := board.at(move)) is None or self.opposes(other):
                yield move
            if other is not None:
                break

        move = pos
        while board.has_pos(move := move.ret(self.is_white).qside()):
            if (other := board.at(move)) is None or self.opposes(other):
                yield move
            if other is not None:
                break

        move = pos
        while board.has_pos(move := move.ret(self.is_white).kside()):
            if (other := board.at(move)) is None or self.opposes(other):
                yield move
            if other is not None:
                break

        move = pos
        while board.has_pos(move := move.adv(self.is_white)):
            if (other := board.at(move)) is None or self.opposes(other):
                yield move
            if other is not None:
                break

        move = pos
        while board.has_pos(move := move.ret(self.is_white)):
            if (other := board.at(move)) is None or self.opposes(other):
                yield move
            if other is not None:
                break

        move = pos
        while board.has_pos(move := move.qside()):
            if (other := board.at(move)) is None or self.opposes(other):
                yield move
            if other is not None:
                break

        move = pos
        while board.has_pos(move := move.kside()):
            if (other := board.at(move)) is None or self.opposes(other):
                yield move
            if other is not None:
                break


class King(Piece):
    def __init__(self, is_white: bool):
        self.is_white = is_white

    def symbol(self):
        return ["♚", "♔"][self.is_white]

    def moves(self, board: Board):
        pos = board.find(self)

        moves = [
            pos.adv(self.is_white),
            pos.adv(self.is_white).qside(),
            pos.qside(),
            pos.ret(self.is_white).qside(),
            pos.ret(self.is_white),
            pos.ret(self.is_white).kside(),
            pos.kside(),
            pos.adv(self.is_white).kside(),
        ]

        for move in moves:
            if not board.has_pos(move):
                continue
            elif (other := board.at(move)) is None or self.opposes(other):
                yield move


@dataclass
class Pos:
    icol: int
    irow: int

    ranks: ClassVar = ["a", "b", "c", "d", "e", "f", "g", "h"]
    files: ClassVar = ["1", "2", "3", "4", "5", "6", "7", "8"]

    def adv(self, is_white: bool):
        irow = self.irow + 1 if is_white else self.irow - 1
        return Pos(self.icol, irow)

    def ret(self, is_white: bool):
        irow = self.irow - 1 if is_white else self.irow + 1
        return Pos(self.icol, irow)

    def kside(self):
        return Pos(self.icol - 1, self.irow)

    def qside(self):
        return Pos(self.icol + 1, self.irow)

    def at_notation(self, note: str):
        if len(note) == 1:
            return self.icol == (self.ranks.index(note) + 1)
        else:
            return self == self.from_notation(note)

    @classmethod
    def from_notation(cls, note: str):
        assert len(note) == 2
        return cls(cls.ranks.index(note[0]) + 1, cls.files.index(note[1]) + 1)


class Board:
    def __init__(self):
        # reverse the order of the rows so that the indexes line up better with
        # rank and file designations
        self.grid: list[list[Piece | None]] = [
            [
                Piece(True)
                for Piece in [Rook, Knight, Bishop, King, Queen, Bishop, Knight, Rook]
            ],
            [Pawn(True) for _ in range(8)],
            [None] * 8,
            [None] * 8,
            [None] * 8,
            [None] * 8,
            [Pawn(False) for _ in range(8)],
            [
                Piece(False)
                for Piece in [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]
            ],
        ]
        self.white_turn = True

    def find(self, piece: Piece):
        """Returns the position tuple (irow, icol) of a piece on the board"""
        for irow, squares in enumerate(self.grid, 1):
            for icol, square in enumerate(squares, 1):
                if piece == square:
                    return Pos(icol, irow)
        raise ValueError("Piece isn't on the board")

    def at(self, pos: Pos):
        """Returns the piece on a particular square position (or `None`)"""
        if not self.has_pos(pos):
            raise IndexError(f"Position isn't on the board")
        return self.grid[pos.irow - 1][pos.icol - 1]

    def has_pos(self, pos: Pos):
        return pos.irow in range(1, 9) and pos.icol in range(1, 9)

    def attackers(self, pos: Pos):
        for irow, squares in enumerate(self.grid, 1):
            for icol, piece in enumerate(squares, 1):
                if piece is None:
                    continue
                elif pos in piece.moves(self):
                    yield (Pos(icol, irow), piece)

    def would_self_check(self, piece: Piece, move: Pos):
        start = self.find(piece)
        capture = self.at(move)
        self.set(start, None)
        self.set(move, piece)

        in_check = False
        for irow, squares in enumerate(self.grid, 1):
            for icol, square in enumerate(squares, 1):
                if not isinstance(square, King) or square.opposes(piece):
                    continue
                elif any(attacker for attacker in self.attackers(Pos(icol, irow))):
                    in_check = True

        self.set(start, piece)
        self.set(move, capture)
        return in_check

    def parse_move(self, move: str):
        check = "[+#]"
        rank = "[1-8]"
        file = "[a-h]"
        piece = "[KQRBN]"

        promote = f"x?(?P<to>{file}[18])=?(?P<piece>{piece}){check}?"
        pawn_move = f"P?(?P<start>{file})?x?(?P<to>{file}{rank}){check}?"
        piece_move = (
            f"(?P<piece>{piece})(?P<start>{file}?{rank}?)x?(?P<to>{file}{rank}){check}?"
        )

        piece_map = {
            "K": King,
            "Q": Queen,
            "R": Rook,
            "B": Bishop,
            "N": Knight,
            "P": Pawn,
        }

        if move == "O-O":
            self.castle(True)
        elif move == "O-O-O":
            self.castle(False)
        elif match := re.match(piece_move, move):
            self.move(
                piece_map[match.group("piece")],
                match.group("to"),
                match.group("start") or None,
            )
        elif match := re.match(pawn_move, move):
            self.move(
                Pawn,
                match.group("to"),
                match.group("start") or None,
            )
        elif match := re.match(promote, move):
            self.promote(match.group("to"), piece_map[match.group("piece")])
        else:
            raise Exception(f"Can't parse move {move!r}")

    def castle(self, short: bool):
        print("CASTLE", short)

    def promote(self, to: str, piece: Type[Piece]):
        print("PROMOTE", to, piece)

    def move(self, type: Type[Piece], to: str, start: str | None):
        assert len(to) == 2

        piece = None
        target = Pos.from_notation(to)
        for pos, attacker in self.attackers(target):
            if attacker.is_white != self.white_turn:
                continue
            elif not isinstance(attacker, type):
                continue
            elif start and not pos.at_notation(start):
                continue
            elif piece is not None:
                raise ValueError("Ambiguous move notation")
            else:
                piece = attacker

        if piece is None:
            raise ValueError("No piece matching move found")
        elif self.would_self_check(piece, target):
            raise ValueError("Move would cause a self check")

        capture = self.at(target)
        self.set(self.find(piece), None)
        self.set(target, piece)
        self.white_turn = not self.white_turn
        return capture

    def set(self, pos: Pos, piece: Piece | None):
        self.grid[pos.irow - 1][pos.icol - 1] = piece

    def to_str(self, highlight: list[Pos] = []):
        output = ["-----" * 8]
        for irow, squares in reversed(list(enumerate(self.grid, 1))):
            output.append("")
            for icol, square in enumerate(squares, 1):
                if square is None and Pos(icol, irow) in highlight:
                    output[-1] += "| () "
                elif square is None:
                    output[-1] += "|    "
                elif Pos(icol, irow) in highlight:
                    output[-1] += f"|({square.symbol()} )"
                else:
                    output[-1] += f"| {square.symbol()}  "
            output[-1] += "|"
            output.append("-----" * 8)
        return "\n".join(output)


class Matchmaker:
    def __init__(self):
        self.games = {}
        self.queue = Queue[Game](10)

    async def find_game(self, player: Player):
        await self.queue.put(Game(player))

        while game := await self.queue.get():
            if game.is_playing(player):
                await self.queue.put(game)
            elif game.awaiting_opponent():
                break

        game.set_opponent(player)
        return game


if __name__ == "__main__":
    board = Board()

    highlight = []
    while True:
        print(board.to_str(highlight))
        print("White to move" if board.white_turn else "Black to move")
        cmd, value = input("pick/move >>> \u001B[K").split(None, 1)
        if cmd == "pick":
            pos = Pos.from_notation(value)
            piece = board.at(Pos.from_notation(value))
            if piece is None:
                highlight = [pos]
            else:
                moves = [
                    move
                    for move in piece.moves(board)
                    if not board.would_self_check(piece, move)
                ]
                highlight = [pos, *moves]
        elif cmd == "move":
            highlight = []
            board.parse_move(value)

        print("\x1B[19A", end="")
