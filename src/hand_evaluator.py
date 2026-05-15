import treys
from treys import Card, Evaluator

class HandEvaluator:
    def __init__(self):
        self.evaluator = Evaluator()

    def evaluate(self, hole_cards, community_cards):
        """
        接收字串列表或 Card 物件列表，回傳手牌強度分數 (越小越強)
        """
        # 轉換為 Card 物件（若為字串）
        if hole_cards and isinstance(hole_cards[0], str):
            hole_cards = [Card.new(c) for c in hole_cards]
        if community_cards and isinstance(community_cards[0], str):
            community_cards = [Card.new(c) for c in community_cards]
        return self.evaluator.evaluate(community_cards, hole_cards)

    def get_hand_class(self, hole_cards, community_cards):
        score = self.evaluate(hole_cards, community_cards)
        return self.evaluator.get_rank_class(score)

    @staticmethod
    def cards_from_str(card_strs):
        return [Card.new(card_str) for card_str in card_strs]

if __name__ == "__main__":
    evaluator = HandEvaluator()
    hole = ['Ah', 'Kh']
    board = ['Qh', 'Jh', 'Th', '2c', '3d']
    score = evaluator.evaluate(hole, board)
    rank_class = evaluator.get_rank_class(hole, board)
    print(f"Score: {score}, Class: {rank_class} ({evaluator.evaluator.class_to_string(rank_class)})")
