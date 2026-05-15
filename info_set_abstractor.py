from enum import IntEnum
import math

class BetBucket(IntEnum):
    """下注尺寸桶化 (4 檔)"""
    SMALL = 1      # 約 25-50% 底池
    MEDIUM = 2     # 約 50-80% 底池
    LARGE = 3      # 約 80-150% 底池
    OVERBET = 4    # 超過 150% 底池 或 All-in

class BoardTexture(IntEnum):
    """公共牌紋理分類 (10 類，基於翻牌後 3 張)"""
    HIGH_DRY = 1        # 高牌乾燥 (如 K72r)
    HIGH_WET = 2        # 高牌濕潤 (如 KQJ兩同花)
    MID_DRY = 3         # 中牌乾燥 (如 963r)
    MID_WET = 4         # 中牌濕潤 (如 987兩同花)
    LOW_DRY = 5         # 低牌乾燥 (如 522r)
    LOW_WET = 6         # 低牌濕潤 (如 654兩同花)
    PAIRED = 7          # 公牌有對子
    MONOTONE = 8        # 三張同花 (Mono-flop)
    TWO_TONE = 9        # 兩張同花 (Two-tone flop)
    RAINBOW = 10        # 彩虹牌面 (Rainbow)

class InfoSetAbstractor:
    """資訊集抽象器：將連續狀態壓縮為有限分類"""

    # ────────── 下注尺寸桶化 ──────────
    @staticmethod
    def discretize_bet(bet_amount, pot_size):
        """
        將下注金額離散化為 4 個桶子。
        回傳 BetBucket 枚舉值。
        """
        if pot_size <= 0:
            return BetBucket.MEDIUM  # 安全預設
        
        fraction = bet_amount / pot_size
        
        if fraction <= 0.5:
            return BetBucket.SMALL
        elif fraction <= 0.8:
            return BetBucket.MEDIUM
        elif fraction <= 1.5:
            return BetBucket.LARGE
        else:
            return BetBucket.OVERBET

    # ────────── 公共牌紋理分類 (僅限翻牌 3 張) ──────────
    @staticmethod
    def classify_board(cards):
        """
        cards: list of treys.Card 物件 (通常 3 張翻牌)
        回傳 BoardTexture 枚舉值。
        """
        from treys import Card
        
        ranks = [Card.get_rank_int(c) for c in cards]
        suits = [Card.get_suit_int(c) for c in cards]
        
        # 1. 有對子？
        if len(set(ranks)) < len(ranks):
            return BoardTexture.PAIRED
        
        # 2. 花色分析
        unique_suits = len(set(suits))
        if unique_suits == 1:
            return BoardTexture.MONOTONE
        elif unique_suits == 2:
            is_two_tone = True  # 兩同花面
        else:
            is_two_tone = False  # 彩虹面
        
        # 3. 高/中/低牌判斷 (以最大牌為基準)
        max_rank = max(ranks)
        # 0=2, ..., 12=Ace
        if max_rank >= 10:   # T, J, Q, K, A
            high = True
        elif max_rank >= 7:  # 7, 8, 9
            high = False
            mid = True
        else:                # 2~6
            high = False
            mid = False
        
        # 4. 連牌度 (濕潤度) 判斷
        sorted_ranks = sorted(ranks)
        gaps = sum(1 for i in range(len(sorted_ranks)-1) 
                   if sorted_ranks[i+1] - sorted_ranks[i] <= 2)
        is_wet = (gaps >= 2) or (max_rank - min(ranks) <= 3)
        
        # 5. 組合分類
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

    # ────────── 完整資訊集編碼 ──────────
    @staticmethod
    def encode(game_round, bet_history, board_cards, pot_size, stack_size):
        """
        將一局遊戲狀態編碼為一個壓縮的資訊集字串。
        格式: "round|bet_bucket|board_texture|stack_to_pot_ratio"
        回傳 str
        """
        # 回合: preflop, flop, turn, river
        round_map = {0: "P", 1: "F", 2: "T", 3: "R"}
        round_str = round_map.get(game_round, "?")
        
        # 下注桶 (取最後一次下注或預設)
        if bet_history and bet_history[-1] > 0 and pot_size > 0:
            last_bet = bet_history[-1]
            bet_bucket = InfoSetAbstractor.discretize_bet(last_bet, pot_size).value
        else:
            bet_bucket = 0  # 無下注
        
        # 公共牌紋理 (只在翻牌後有意義)
        if len(board_cards) >= 3:
            texture = InfoSetAbstractor.classify_board(board_cards[:3]).value
        else:
            texture = 0  # 翻牌前無紋理
        
        # 籌碼深度 (stack-to-pot ratio)
        if pot_size > 0:
            spr = int(stack_size / pot_size)
        else:
            spr = 999
        
        return f"{round_str}|{bet_bucket}|{texture}|{spr}"

# ────────── 快速測試 ──────────
if __name__ == "__main__":
    from treys import Card
    
    abstractor = InfoSetAbstractor()
    
    # 測試 1: 下注桶化
    print("=== 下注桶化測試 ===")
    tests = [(25, 100), (60, 100), (100, 100), (200, 100)]
    for bet, pot in tests:
        bucket = abstractor.discretize_bet(bet, pot)
        print(f"  下注 {bet} / 底池 {pot} → {bucket.name}")
    
    # 測試 2: 公共牌紋理
    print("\n=== 公共牌紋理測試 ===")
    board1 = [Card.new(s) for s in ['Ks', '7d', '2c']]  # 高牌乾燥
    board2 = [Card.new(s) for s in ['Kh', 'Qh', 'Jh']]  # 高牌濕潤同花
    board3 = [Card.new(s) for s in ['5s', '5d', '2c']]  # 有對子
    for board in [board1, board2, board3]:
        texture = abstractor.classify_board(board)
        print(f"  {[Card.int_to_pretty_str(c) for c in board]} → {texture.name}")
    
    # 測試 3: 完整編碼
    print("\n=== 完整資訊集編碼 ===")
    info_set = abstractor.encode(
        game_round=1,           # flop
        bet_history=[50],       # 上次下注 50
        board_cards=board1,
        pot_size=150,
        stack_size=800
    )
    print(f"  資訊集: {info_set}")
