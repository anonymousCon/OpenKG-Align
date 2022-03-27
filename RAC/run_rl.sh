#!/bin/sh

# _degree_ada linearly adapting the avg weights
# avg
# our

# SRPRS/en_fr
CUDA_VISIBLE_DEVICES=1 python ls_mab.py --log gcnalign \
                                    --seed 2020\
                                    --data_dir "data/fb_dbp" \
                                    --rate 0.3 \
                                    --epoch 300 \
                                    --check 300 \
                                    --update 10 \
                                    --train_batch_size -1 \
                                    --encoder "GCN-Align" \
                                    --encoder1 "APP" \
                                    --hiddens "100,100,100" \
                                    --heads "2,2" \
                                    --decoder "Align" \
                                    --sampling "N" \
                                    --k "25" \
                                    --margin "1" \
                                    --alpha "1" \
                                    --feat_drop 0.0 \
                                    --lr 0.002\
                                    --train_dist "euclidean" \
                                    --test_dist "euclidean"