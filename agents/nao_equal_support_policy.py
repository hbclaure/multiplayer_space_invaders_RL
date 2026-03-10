# agents/nao_equal_support_policy.py

class EqualSupport:
    #Nao Switches support every 1000 frames
    def __init__(self):
        self.support = 1

    def update_support(self, frame,scores, support_count_history):
        if frame % 1000 == 0:
            self.support = 1 if self.support == 0 else 0
