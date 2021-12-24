""" For visualizing stuff """

import matplotlib.pyplot as plt
import seaborn as sns

def vis_token_counts(t_c:list):
    plt.figure(figsize=(14,10))
    sns.histplot(t_c)
    plt.title("Token counts per sample")
    plt.show()
