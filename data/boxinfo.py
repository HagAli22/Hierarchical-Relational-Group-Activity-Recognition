import os
import sys

sys.path.append('D:\VSCODE\GNN')


class BoxInfo:
    def __init__(self, line):
        #line like : 0 361 469 413 569 24735 0 1 0 setting
        words = line.split()
        self.category = words.pop()
        words = [int(string) for string in words]
        self.player_ID = words[0]
        del words[0]

        x1, y1, x2, y2, frame_ID, lost, grouping, generated = words
        self.box = x1, y1, x2, y2
        self.frame_ID = frame_ID
        self.lost = lost
        self.grouping = grouping
        self.generated = generated
    
    sys.modules['boxinfo'] = sys.modules[__name__]   