#!/usr/bin/env python

from datagen import run_datagen
from features import run_feature_generation
from model import run_model_training


def run():
    # run_datagen()
    run_feature_generation()
    run_model_training()


if __name__ == "__main__":
    run()
