#!/usr/bin/env python3

import sys
from pipeline import *

def main(infile, 
         model_path, 
         data_outdir, 
         model_outdir, 
         max_length, 
         batch_size, 
         num_epochs, 
         learning_rate):
    preprocessing(infile, data_outdir)
    train_iter, valid_iter, test_iter = tokenizer(data_outdir, model_path, max_length, batch_size)
    training(model_path, train_iter, valid_iter, model_outdir, num_epochs, learning_rate)
    evaluation(model_path, model_outdir, test_iter)

main(infile=sys.argv[1], 
     model_path=sys.argv[2], 
     data_outdir=sys.argv[3], 
     model_outdir=sys.argv[4], 
     max_length=int(sys.argv[5]), 
     batch_size=int(sys.argv[6]),
     num_epochs=int(sys.argv[7]), 
     learning_rate=float(sys.argv[8]))

