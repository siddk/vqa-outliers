# Mind Your Outliers!

> [**Mind Your Outliers! Investigating the Negative Impact of Outliers on Active Learning for Visual Question Answering**](https://arxiv.org/abs/2107.02331)  
> Siddharth Karamcheti, Ranjay Krishna, Li Fei-Fei, Christopher D. Manning  
> Annual Meeting for the Association of Computational Linguistics (ACL-IJCNLP) 2021.

Code & Experiments for training various models and performing active learning on a variety of different VQA datasets
and splits. Additional code for creating and visualizing dataset maps, for qualitative analysis!

If there are any trained models you want access to that aren't easy for you to train, please let me know and I will
do my best to get them to you. Unfortunately finding a hosting solution for 1.8TB of checkpoints hasn't been easy 
:sweat_smile:.

---

## Quickstart

Clones `vqa-outliers` to the current working directory, then walks through dependency setup, mostly leveraging the 
`environments/environment-<arch>` files. Assumes `conda` is installed locally (and is on your path!). [Follow the 
directions here to install `conda` (Anaconda or Miniconda) 
if not](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

We provide two installation directions -- one set of instructions for CUDA-equipped machines running Linux w/ GPUs 
(for training), and another for CPU-only machines (e.g., MacOS, Linux) geared towards local development and in case
GPUs are not available.

The existing GPU YAML File is geared for CUDA 11.0 -- if you have older GPUs, file an issue, and I'll create an 
appropriate conda configuration!

### Setup Instructions

```bash
# Clone `vqa-outliers` Repository and run Conda Setup
git clone https://github.com/siddk/vqa-outliers.git
cd vqa-outliers

# Ensure you're using the appropriate hardware config!
conda env create -f environments/environment-{cpu, gpu}.yaml
conda activate vqa-outliers
``` 

---

## Usage

The following section walks through downloading all the necessary data (be warned -- it's a lot!) and running both the
various active learning strategies on the given VQA datasets, as well as the code for generating Dataset Maps over the 
full dataset, and visualizing active learning acquisitions relative to those maps.

*Note: This is going to require several hundred GB of disk space -- for targeted experiments, feel free to file
an issue and I can point you to what you need!*

### Downloading Data

We have dependencies on a few datasets, some pretrained word vectors (GloVe), and a pretrained multimodal model 
(LXMERT), though not the one commonly released in HuggingFace Transformers. To download all dependencies, use the 
following commands *from the root of this repository* (in general, run everything from repository root!).

```bash
# Note: All the following will create/write to the directory data/ in the current repository -- feel free to change!

# GloVe Vectors
./scripts/download/glove.sh

# Download LXMERT Checkpoint (no-QA Pretraining)
./scripts/download/lxmert.sh

# Download VQA-2 Dataset (Entire Thing -- Questions, Raw Images, BottomUp Object Features)!
./scripts/download/vqa2.sh

# Download GQA Dataset (Entire Thing -- Questions, Raw Images, BottomUp Object Features)!
./scripts/download/gqa.sh
```

### Additional Preprocessing

Many of the models we evaluate in this work use the object-based BottomUp-TopDown Attention Features -- however, our
Grid Logistic Regression and LSTM-CNN Baseline both use dense ResNet-101 Features of the images. We extract these from 
the raw images ourselves as follows (again, this will take a ton of disk space):

```bash
# Note: GPU Recommended for Faster Extraction

# Extract VQA-2 Grid Features
python scripts/extract.py --dataset vqa2 --images data/VQA2-Images --spatial data/VQA2-Spatials

# Extract GQA Grid Features
python scripts/extract.py --dataset gqa --images data/GQA-Images --spatial data/GQA-Spatials
```

### Running Active Learning

Running Active Learning is a simple matter of using the script `active.py` in the root of this directory. This script
is able to reproduce every experiment from the paper, and allows you to specify the following:
- Dataset in `< vqa2 | gqa >`
- Split in `< all | sports | food >` (for VQA-2) and `all` for GQA
- Model (mode) in `< glreg | olreg | cnn | butd | lxmert >` (Both Logistic Regression Models, LSTM-CNN, BottomUp-TopDown, and LXMERT, respectively)
- Active Learning Strategy in `< baseline | least-conf | entropy | mc-entropy | mc-bald | coreset-{fused, language, vision} >` following the paper.
- Size of Seed Set (burn, for burn-in) in `< p05 | p10 | p25 | p50 >` where each denotes percentage of full-dataset to use as seed set. 

For example, to run the BottomUp-TopDown Attention Model (`butd`) with the VQA-2 Sports Dataset, with Bayesian Active Learning by Disagreement,
with a seed set that's 10\% the size of the original dataset, use the following:

```bash
# Note: If GPU available (recommended), pass --gpus 1 as well!
python active.py --dataset vqa2 --split sports --mode butd --burn p10 --strategy mc-bald
```

File an issue if you run into trouble!

### Creating Dataset Maps

Creating a Dataset Map entails training a model on an entire dataset, while maintaining statistics on a **per-example**
basis, over the course of training. To train models and dump these statistics, use the top-level file `cartograph.py` as
follows (again, for the BottomUp-TopDown Model, on VQA2-Sports):

```bash
python cartograph.py --dataset vqa2 --split sports --mode butd
```

Once you've trained a model and generated the necessary statistics, you can plot the corresponding map using
the top-level file `chart.py` as follows:

```bash
# Note: `map` mode only generates the dataset map... to generate acquisition plots, see below!
python chart.py --mode map --dataset vqa2 --split sports --model butd
```

Note that Dataset Maps are generated *per-dataset, per-model*!

### Visualizing Acquisitions

To visualize the acquisitions of a given active learning strategy relative to a given dataset map (the bar graphs from
our paper), you can run the following (again, with our running example, but works for any combination):

```bash
python chart.py --mode acquisitions --dataset vqa2 --split sports --model butd --burn p10 --strategies mc-bald
```

Note that the script `chart.py` defaults to plotting acquisitions for *all* active learning strategies -- either make
sure to run these out for the configuration you want, or provide the appropriate arguments!

### Ablating Outliers

Finally, to run the Outlier Ablation experiments for a given model/active learning strategy, take the following steps:
- Identify the different "frontiers" of examples (different difficulty classes) by using `scripts/frontier.py`
- Once this file has been generated, run `active.py` with the special flag `--dataset vqa2-frontier` and the arbitrary strategies you care about.
- Sit back, examine the results, and get excited!

Concretely, you can generate the `frontier` files for a BottomUp-TopDown Attention Model as follows:

```bash
python scripts/frontier.py --model butd
```

Any other model would also work -- *just make sure you've generated the map via `cartograph.py` first!*

---

## Results

We present the full set of results from the paper (and the additional results from the supplement) in the 
`visualizations/` directory. The sub-directory `active-learning` shows performance vs. samples for various splits of 
strategies (visualizing all on the same plot is a bit taxing), while the sub-directory `acquisitions` has both the
dataset maps and corresponding acquisitions per strategy!

---

## Start-Up (from Scratch)

Use these commands if you're starting a repository from scratch (this shouldn't be necessary to use/build off of this
code, but I like to keep this in the README in case things break in the future). Generally, you should be fine with
the "Usage" section above!

### Linux w/ GPU & CUDA 11.0

```bash
# Create Python Environment (assumes Anaconda -- replace with package manager of choice!)
conda create --name vqa-outliers python=3.8
conda activate vqa-outliers
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
conda install ipython jupyter
conda install pytorch-lightning -c conda-forge

pip install typed-argument-parser h5py opencv-python matplotlib annoy seaborn spacy scipy transformers scikit-learn
```

### Mac OS & Linux (CPU)

```bash
# Create Python Environment (assumes Anaconda -- replace with package manager of choice!)
conda create --name vqa-outliers python=3.8
conda activate vqa-outliers
conda install pytorch torchvision torchaudio -c pytorch
conda install ipython jupyter
conda install pytorch-lightning -c conda-forge

pip install typed-argument-parser h5py opencv-python matplotlib annoy seaborn spacy scipy transformers scikit-learn
```

---

### Note

We are committed to maintaining this repository for the community. We did port this code up to latest
versions of PyTorch-Lightning and PyTorch, so there may be small incompatibilities we didn't catch when testing -- 
please feel free to open an issue if you run into problems, and I will respond within 24 hours. If urgent, please shoot
me an email at `skaramcheti@cs.stanford.edu` with "VQA-Outliers Code" in the Subject line and I'll be happy to help!
