import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, scores2, mean_scores2):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, color='r', linewidth=3)
    plt.plot(mean_scores, color='r', linestyle=':')
    plt.plot(scores2, color='g', linewidth=3)
    plt.plot(mean_scores2, color='g', linestyle=':')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.text(len(scores2)-1, scores2[-1], str(scores2[-1]))
    plt.text(len(mean_scores2)-1, mean_scores2[-1], str(mean_scores2[-1]))
    plt.show(block=False)
    plt.pause(.1)

def plot1(scores, mean_scores ):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, color='r', linewidth=3)
    plt.plot(mean_scores, color='r', linestyle=':')

    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    plt.show(block=False)
    plt.pause(.1)

def plot2( scores2, mean_scores2):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores2, color='g', linewidth=3)
    plt.plot(mean_scores2, color='g', linestyle=':')
    plt.ylim(ymin=0)

    plt.text(len(scores2)-1, scores2[-1], str(scores2[-1]))
    plt.text(len(mean_scores2)-1, mean_scores2[-1], str(mean_scores2[-1]))
    plt.show(block=False)
    plt.pause(.1)