#!/usr/bin/env bash
# gqa.sh
#   Download GQA Dataset and Image Features (Spatial & Object Features).

# Setup
mkdir -p data/GQA-Features
mkdir -p data/GQA-Questions

# Get GQA Features
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/vg_gqa_obj36.zip
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/gqa_testdev_obj36.zip
unzip vg_gqa_obj36.zip
unzip gqa_testdev_obj36.zip

# Clean Up
rm vg_gqa_obj36.zip
rm gqa_testdev_obj36.zip
mv vg_gqa_imgfeat/vg_gqa_obj36.tsv data/GQA-Features
mv vg_gqa_imgfeat/gqa_testdev_obj36.tsv data/GQA-Features
rm -r vg_gqa_imgfeat

# Get GQA Questions --> Delete Unbalanced Dataset, Irrelevant Files (save storage)
wget https://nlp.stanford.edu/data/gqa/questions1.3.zip
unzip questions1.3.zip

# Cleanup -- Focus is only on Balanced Subset
rm questions1.3.zip
mv *balanced_questions.json data/GQA-Questions
rm challenge_all_questions.json
rm submission_all_questions.json
rm test_all_questions.json
rm val_all_questions.json
rm -rf train_all_questions
rm readme.txt

# Get GQA Images
wget https://nlp.stanford.edu/data/gqa/images.zip
unzip images.zip

# Cleanup
rm images.zip
mv images data/GQA-Images
