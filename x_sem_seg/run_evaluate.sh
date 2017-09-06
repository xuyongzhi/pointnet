#!/bin/bash

LOG_DIR="log6"
echo "LOG_DIR=$LOG_DIR"
## data/stanford_indoor3d/Area_6_pantry_1.npy
python batch_inference.py --model_path $LOG_DIR/model.ckpt --dump_dir $LOG_DIR/dump_eval --output_filelist $LOG_DIR/output_filelist_eva.txt --room_data_filelist meta/area6_data_label_1.txt --visu

#python batch_inference.py --model_path log6/model.ckpt --dump_dir log6/dump_eva_spl --output_filelist log6/output_filelist_eva_spl.txt --room_data_filelist meta/area6_data_label_split.txt --visu

#python batch_inference.py --model_path log6/model.ckpt --dump_dir log6/split_dump --output_filelist log6/split_output_filelist.txt --room_data_filelist meta/area6_data_split.txt --visu


## data/Stanford3dDataset_v1.2_Aligned_Version/Area_6/conferenceRoom_1/conferenceRoom_1.txt
#python batch_inference.py --model_path log6/model.ckpt --dump_dir log6/dump --output_filelist log6/output_filelist.txt --room_data_filelist meta/area6_data.txt --visu


## ETH_reduced_test_split.txt
#python batch_inference.py --model_path log6/model.ckpt --dump_dir log6/ETH/reduced_split/dump --output_filelist log6/ETH/reduced_split/output_filelist.txt --room_data_filelist meta/ETH_reduced_test_split.txt --visu --del_column 3
