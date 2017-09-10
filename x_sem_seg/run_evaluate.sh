#!/bin/bash

#LOG_DIR="log5_4096_bs32"
LOG_DIR="LOG_QI/log1"
echo "LOG_DIR=$LOG_DIR"

#python train_sorted.py --only_evaluate --log_dir $LOG_DIR  --batch_size 4 --num_point 4096 --eval_data_rate 1   --all_fn_glob stanford_indoor3d_normedh5_stride_0.5_step_1_4096/Area_6_office_22     --visu Area_6_office_22









python batch_inference.py --model_path  $LOG_DIR/model.ckpt --dump_dir  $LOG_DIR/dump_eva_spl --output_filelist $LOG_DIR/output_filelist_eva_1.txt --room_data_filelist meta/area6_data_label_1.txt --visu

#python batch_inference.py --model_path log6/model.ckpt --dump_dir log6/split_dump --output_filelist log6/split_output_filelist.txt --room_data_filelist meta/area6_data_split.txt --visu


## data/Stanford3dDataset_v1.2_Aligned_Version/Area_6/conferenceRoom_1/conferenceRoom_1.txt
#python batch_inference.py --model_path log6/model.ckpt --dump_dir log6/dump --output_filelist log6/output_filelist.txt --room_data_filelist meta/area6_data.txt --visu


## ETH_reduced_test_split.txt
#python batch_inference.py --model_path log6/model.ckpt --dump_dir log6/ETH/reduced_split/dump --output_filelist log6/ETH/reduced_split/output_filelist.txt --room_data_filelist meta/ETH_reduced_test_split.txt --visu --del_column 3











#  ('/home/x/Research/pointnet/data/stanford_indoor3d/Area_6_office_22.npy', 'LOG_QI/log1/dump_eva_spl/Area_6_office_22_pred.txt')
#  file_size= 12
#  eval mean loss: 0.175759
#  eval accuracy: 0.928446
#  all room eval accuracy: 0.928446
