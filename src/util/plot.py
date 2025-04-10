import matplotlib
from matplotlib import font_manager, pyplot as plt
import os

def init_matplotlib(latex=True, font_size=15):
    ttf_var = 'MATPLOTLIB_TTF'

    if ttf_var in os.environ:
        font_path = os.environ[ttf_var]

        font_manager.fontManager.addfont(font_path)

        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = prop.get_name()

    matplotlib.rcParams.update({'font.size': font_size})

    if latex:
        matplotlib.rcParams['text.usetex'] = True
        latex_preamble = r'''
            \usepackage{tgpagella} % text only
            \usepackage{mathpazo}  % math & text
        '''

        # Set the LaTeX preamble in Matplotlib
        plt.rcParams['text.latex.preamble'] = latex_preamble
