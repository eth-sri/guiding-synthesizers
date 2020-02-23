#!/bin/bash
# Script to extract data and models

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Extract Models
wget http://files.srl.inf.ethz.ch/data/guiding_synthesizers/models.tar.gz
mv models.tar.gz $DIR/core/models.tar.gz
tar -C $DIR/core -xvzf $DIR/core/models.tar.gz

# Extract Data
tar -C $DIR/dataset/data/ -xvzf $DIR/dataset/data/ablation_dataset.tar.gz
tar -C $DIR/dataset/data/ -xvzf $DIR/dataset/data/ds.tar.gz
tar -C $DIR/dataset/data/ -xvzf $DIR/dataset/data/dsplus.tar.gz
