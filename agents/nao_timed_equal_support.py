# agents/nao_timed_equal_support_policy.py

class TimedEqualSupport:
    #Nao Switches support at the halfway point or after  5000 frames
    def __init__(self):
        self.support = 1

    def update_support(self, frame,scores,support_count_history):
        if frame <= 3600:
            self.support = 1 
        else:
            self.support = 0
