import matplotlib
import matplotlib as mpl
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
    else:
        mpl.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"],  # or "Times"
            #"axes.labelsize": 10,
            #"font.size": 10,
            #"legend.fontsize": 9,
            #"xtick.labelsize": 9,
            #"ytick.labelsize": 9,
        })

    matplotlib.rcParams.update({'font.size': font_size})

    if latex:
        matplotlib.rcParams['text.usetex'] = True
        latex_preamble = r'''
            \usepackage{tgpagella} % text only
            \usepackage{mathpazo}  % math & text
            \usepackage{times}
            \usepackage{amsmath}
            \usepackage{amssymb}
            \usepackage{glossaries}
            
            \newacronym{mamut}{MAMUT}{Math Mutator}


\newacronym[plural=NNs, firstplural=Neural Networks (NN)]{nn}{NN}{Neural Network}
\newacronym[plural=ANNs, firstplural=Artificial Neural Networks (ANN)]{ann}{ANN}{Artificial Neural Network}
\newacronym[plural=RNNs, firstplural=Recurrent Neural Networks (RNN)]{rnn}{RNN}{Recurrent Neural Network}
\newacronym{cnn}{CNN}{Convolutional Neural Network}
\newacronym{gan}{GAN}{Generative Adversarial Network}
\newacronym{gpt}{GPT}{Generative Pre-trained Transformer}

\newacronym{nlp}{NLP}{Natural Language Processing}

\newacronym{arqmath}{\mbox{ARQMath}}{Answer Retrieval for Questions on Math}
\newacronym{amps}{AMPS}{Auxiliary Mathematics Problems and Solutions}
\newacronym{bert}{BERT}{Bidirectional Encoder Representations from Transformers}
\newacronym{albert}{ALBERT}{A Lite \acrshort{bert}}
\newacronym{roberta}{RoBERTa}{Robustly Optimized \acrshort{bert} Approach}
\newacronym{deberta}{DeBERTa}{Decoding-Enhanced \acrshort{bert} with disentangled Attention}
\newacronym{ir}{IR}{Information Retrieval}
\newacronym{nmft}{NMFT}{Named Mathematical Formula Templates}
\newacronym{smf}{SMF}{Structured Mathematical Formulas}
\newacronym{mir}{MIR}{Mathematical Information Retrieval}
\newacronym{mlm}{MLM}{Masked Language Modeling}
\newacronym{nsp}{NSP}{Next Sentence Prediction}
\newacronym{clm}{CLM}{Causal Language Modeling}
\newacronym{rtd}{RTD}{Replaced Token Detection}
\newacronym{msas}{MSAS}{Mathematical Structure Attention Score}
\newacronym{pca}{PCA}{Principle Component Analysis}


\newacronym{dcg}{DCG}{Discounted Cumulative Gain}
\newacronym{ndcg}{nDCG}{Normalized Discounted Cumulative Gain}
\newacronym{ndcg'}{nDCG'}{normalized Discounted Cumulative Gain when assessed on only judged documents}
\newacronym{map}{mAP}{Mean Average Precision}
\newacronym{ap}{AP}{Average Precision}
\newacronym[plural=\text{p@}k]{patk}{p@$k$}{Precicion at $k$}

\newacronym{mf}{MF}{Mathematical Formulas}
\newacronym{mt}{MT}{Mathematical Texts} 
\newacronym{nmf}{NMF}{Named Mathematical Formulas} 
\newacronym{mfr}{MFR}{Mathematical Formula Retrieval}

\newacronym{mfpt}{\acrshort{mf}-PT}{\acrshort{mf} Pre-Training}
\newacronym{mtpt}{\acrshort{mt}-PT}{\acrshort{mt} Pre-Training}
\newacronym{nmfpt}{\acrshort{nmf-PT}}{\acrshort{nmf} Pre-Training}
\newacronym{mfrpt}{\acrshort{mfr}-PT}{\acrshort{mfr} Pre-Training}
\newacronym{nmfft}{\acrshort{nmf}-FT}{\acrshort{nmf} Fine-Tuning}
\newacronym{mfrft}{\acrshort{mfr}-FT}{\acrshort{mfr} Fine-Tuning}

\newacronym{evg}{\mbox{EquVG}}{Equivalent Version Generation}
\newacronym{fvg}{\mbox{FalseVG}}{Falsified Version Generation}

            
\newcommand{\bertbase}[0]{BERT}

\newcommand{\mathPretrainedBert}{MP\acrshort{bert}} % AnReu/math\_pretrained\_bert
%\newcommand{\mathPretrainedBertTF}{$\text{\mathPretrainedBert}_{\scriptscriptstyle{\text{constant falses}}}$}
%\newcommand{\mathPretrainedBertNCF}{$\text{\mathPretrainedBert}_{\scriptscriptstyle{\text{random falses}}}$}
\newcommand{\mathPretrainedBertTF}{\mathPretrainedBert-constant-falses}
\newcommand{\mathPretrainedBertNCF}{\mathPretrainedBert-random-falses}


\newcommand{\mamutbert}{\acrshort{mamut}-\acrshort{bert}}
\newcommand{\mamutmathbert}{\acrshort{mamut}-\mathbert}
\newcommand{\mamutmpbert}{\acrshort{mamut}-\mathPretrainedBert}

\newcommand{\mamutbertmfmt}{\mamutbert-\acrshort{mlm}}
\newcommand{\mamutmpbertmfmt}{\mamutmpbert-\acrshort{mlm}}
\newcommand{\mamutmathbertmfmt}{\mamutmathbert-\acrshort{mlm}}

\newcommand{\mathbert}{MathBERT} % tbs17/MathBERT
\newcommand{\mathbertcustom}{\mathbert-custom}
\newcommand{\mathBerta}{MathBERTa}
\newcommand{\scibert}{SciBERT}
        '''

        # Set the LaTeX preamble in Matplotlib
        plt.rcParams['text.latex.preamble'] = latex_preamble
