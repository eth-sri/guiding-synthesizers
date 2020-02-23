#!/bin/bash
# Script to extract data and models

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
tar -C $DIR/core -xvzf $DIR/core/saved_modelsPaper.tar.gz
tar -C $DIR/dataset/data/ -xvzf $DIR/dataset/data/ablation_dataset.tar.gz
tar -C $DIR/dataset/data/ -xvzf $DIR/dataset/data/ds.tar.gz
tar -C $DIR/dataset/data/ -xvzf $DIR/dataset/data/dsplus.tar.gz
