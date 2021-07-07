#!/usr/bin/env bash
# lxmert.sh
#   Download LXMERT Checkpoint WITHOUT pretraining directly on full VQA-2/GQA train sets (for fair comparison!).
#   Checkpoint provided courtesy of Hao Tan (@airsplay).

# Setup
mkdir -p data/LXMERT

# Fetch Checkpoint
wget -P data/LXMERT http://nlp.cs.unc.edu/data/lxrt_noqa/Epoch19_LXRT.pth
