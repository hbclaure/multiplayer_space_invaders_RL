# agents/nao_partial_support_human.py

class partialSupportHuman:
    #Nao support human for the first 1/4 of the game
    def __init__(self):
        self.support = 1

    def update_support(self, frame,scores,support_count_history):
        if frame <= 1800:
            self.support = 1 
        else:
            self.support = 0
