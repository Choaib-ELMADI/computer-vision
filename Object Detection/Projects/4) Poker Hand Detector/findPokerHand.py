def findPokerHand(hand):
    pokerHandRanks = {
        10: "Royal Flush",
        9: "Straight Flush",
        8: "Four of a Kind",
        7: "Full House",
        6: "Flush",
        5: "Straight",
        4: "Three of a Kind",
        3: "Two Pair",
        2: "Pair",
        1: "High Card",
    }
    ranks = []
    suits = []
    possibleRanks = []

    for card in hand:
        if len(card) == 3:
            rank = card[0:2]
        else:
            rank = card[0]

        if rank == "A":
            rank = 14
        elif rank == "K":
            rank = 13
        elif rank == "Q":
            rank = 12
        elif rank == "J":
            rank = 11

        suit = card[-1]

        ranks.append(int(rank))
        suits.append(suit)

    sortedRanks = sorted(ranks)

    #! Royal Flush, Straight Flush, Flush
    if suits.count(suits[0]) == len(suits):
        if [10, 11, 12, 13, 14] == sortedRanks:
            possibleRanks.append(10)
        elif sortedRanks[0] == sortedRanks[4] - 4:
            possibleRanks.append(9)
        else:
            possibleRanks.append(6)

    ranksUniqueVals = list(set(sortedRanks))
    if len(ranksUniqueVals) == 2:
        for val in ranksUniqueVals:
            #! Four of a Kind
            if sortedRanks.count(val) == 4:
                possibleRanks.append(8)
            #! Full House
            else:
                possibleRanks.append(7)
    elif len(ranksUniqueVals) == 3:
        for val in ranksUniqueVals:
            #! Three of a Kind
            if sortedRanks.count(val) == 3:
                possibleRanks.append(4)
            #! Two Pair
            elif sortedRanks.count(val) == 2:
                possibleRanks.append(3)
    #! Pair
    elif len(ranksUniqueVals) == 4:
        possibleRanks.append(2)

    #! Straight
    if all(
        sortedRanks[i] == sortedRanks[i - 1] + 1 for i in range(1, len(sortedRanks))
    ):
        possibleRanks.append(5)

    #! High Card
    if not possibleRanks:
        possibleRanks.append(1)

    output = pokerHandRanks[max(possibleRanks)]
    # print(hand, " --> ", output)
    return output


if __name__ == "__main__":
    findPokerHand(["KH", "AH", "QH", "JH", "10H"])  # ! ---> Royal Flush
    findPokerHand(["QC", "JC", "10C", "9C", "8C"])  # * ---> Straight Flush
    findPokerHand(["5C", "5S", "5H", "5D", "QH"])  # ! ----> Four of a Kind
    findPokerHand(["2H", "2D", "2S", "10H", "10C"])  # * --> Full House
    findPokerHand(["2D", "KD", "7D", "6D", "5D"])  # ! ----> Flush
    findPokerHand(["JC", "10H", "9C", "8C", "7D"])  # * ---> Straight
    findPokerHand(["10H", "10C", "10D", "2D", "5S"])  # ! -> Three of a Kind
    findPokerHand(["KD", "KH", "5C", "5S", "6D"])  # * ----> Two Pair
    findPokerHand(["2D", "2S", "9C", "KD", "10C"])  # ! ---> Pair
    findPokerHand(["KD", "5H", "2D", "10C", "JH"])  # * ---> High Card
