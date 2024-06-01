import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualizer():
    def __init__(self, style='seaborn', palette='viridis'):
        plt.style.use(style)
        self.palette = palette
    