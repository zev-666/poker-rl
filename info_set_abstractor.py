from enum import IntEnum
import math
from treys import Card

class BetBucket(IntEnum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3
    OVERBET = 4

class BoardTexture(IntEnum):
    HIGH_DRY = 1
    HIGH_WET = 2
    MID_DRY = 3
    MID_WET = 4
    LOW_DRY = 5
    LOW_WET = 6
    PAIRED = 7
    MONOTONE = 8
    TWO_TONE = 9
    RAINBOW = 10

class InfoSetAbstractor:
    @staticmethod
    def discretize_bet(bet_amount, pot_size):
        if pot_size <= 0:
            return BetBucket.MEDIUM
        fraction = bet_amount / pot_size
        if fraction <= 0.5:
            return BetBucket.SMALL
        elif fraction <= 0.8:
            return BetBucket.MEDIUM
        elif fraction <= 1.5:
            return BetBucket.LARGE
        else:
            return BetBucket.OVERBET

    @staticmethod
    def classify_board(cards):
        # 支援字串列表 → 自動轉成 Card 物件
        if cards and isinstance(cards[0], str):
            cards = [Card.new(c) for c in cards]
        ranks = [Card.get_rank_int(c) for c in cards]
        suits = [Card.get_suit_int(c) for c in cards]
        if len(set(ranks)) < len(ranks):
            return BoardTexture.PAIRED
        unique_suits = len(set(suits))
        if unique_suits == 1:
            return BoardTexture.MONOTONE
        elif unique_suits == 2:
            is_two_tone = True
        else:
            is_two_tone = False
        max_rank = max(ranks)
        if max_rank >= 10:
            high = True
        elif max_rank >= 7:
            high = False
            mid = True
        else:
            high = False
            mid = False
        sorted_ranks = sorted(ranks)
        gaps = sum(1 for i in range(len(sorted_ranks)-1)
                   if sorted_ranks[i+1] - sorted_ranks[i] <= 2)
        is_wet = (gaps >= 2) or (max_rank - min(ranks) <= 3)
        if high:
            if is_wet or is_two_tone:
                return BoardTexture.HIGH_WET
            else:
                return BoardTexture.HIGH_DRY
        elif mid:
            if is_wet or is_two_tone:
                return BoardTexture.MID_WET
            else:
                return BoardTexture.MID_DRY
        else:
            if is_wet or is_two_tone:
                return BoardTexture.LOW_WET
            else:
                return BoardTexture.LOW_DRY

    @staticmethod
    def encode(game_round, bet_history, board_cards, pot_size, stack_size):
        # 接受字串列表 → 轉成 Card 物件供 classify_board 使用
        if board_cards and isinstance(board_cards[0], str):
            board_cards = [Card.new(c) for c in board_cards]
        round_map = {0: "P", 1: "F", 2: "T", 3: "R"}
        round_str = round_map.get(game_round, "?")
        if bet_history and bet_history[-1] > 0 and pot_size > 0:
            last_bet = bet_history[-1]
            bet_bucket = InfoSetAbstractor.discretize_bet(last_bet, pot_size).value
        else:
            bet_bucket = 0
        if len(board_cards) >= 3:
            texture = InfoSetAbstractor.classify_board(board_cards[:3]).value
        else:
            texture = 0
        if pot_size > 0:
            spr = int(stack_size / pot_size)
        else:
            spr = 999
        return f"{round_str}|{bet_bucket}|{texture}|{spr}"

if __name__ == "__main__":
    abstractor = InfoSetAbstractor()
    print("=== 下注桶化測試 ===")
    tests = [(25, 100), (60, 100), (100, 100), (200, 100)]
    for bet, pot in tests:
        bucket = abstractor.discretize_bet(bet, pot)
        print(f"  下注 {bet} / 底池 {pot} → {bucket.name}")
    print("\n=== 公共牌紋理測試 ===")
    board1 = ['Ks', '7d', '2c']
    board2 = ['Kh', 'Qh', 'Jh']
    board3 = ['5s', '5d', '2c']
    for board in [board1, board2, board3]:
        texture = abstractor.classify_board(board)
        print(f"  {board} → {texture.name}")
    print("\n=== 完整資訊集編碼 ===")
    info_set = abstractor.encode(1, [50], board1, 150, 800)
    print(f"  資訊集: {info_set}")
