import treys
from treys import Card, Evaluator

class HandEvaluator:
    def __init__(self):
        self.evaluator = Evaluator()

    def evaluate(self, hole_cards, community_cards):
        return self.evaluator.evaluate(community_cards, hole_cards)

    def get_hand_class(self, hole_cards, community_cards):
        score = self.evaluate(hole_cards, community_cards)
        return self.evaluator.get_rank_class(score)

    @staticmethod
    def cards_from_str(card_strs):
        return [Card.new(card_str) for card_str in card_strs]

if __name__ == "__main__":
    evaluator = HandEvaluator()
    hole = HandEvaluator.cards_from_str(['Ah', 'Kh'])
    board = HandEvaluator.cards_from_str(['Qh', 'Jh', 'Th', '2c', '3d'])
    score = evaluator.evaluate(hole, board)
    rank_class = evaluator.get_hand_class(hole, board)
    print(f"Score: {score}, Class: {rank_class} ({evaluator.evaluator.class_to_string(rank_class)})")
