#!/bin/bash

LOG_DIR="log1"
python batch_inference.py --model_path $LOG_DIR/model.ckpt --dump_dir $LOG_DIR/dump --output_filelist $LOG_DIR/output_filelist.txt --room_data_filelist meta/area6_data_label_1.txt --visu
