#!/bin/bash

LOG_DIR="log6_org_OK"
python batch_inference.py --model_path $LOG_DIR/model.ckpt --dump_dir $LOG_DIR/dump --output_filelist $LOG_DIR/output_filelist.txt --room_data_filelist meta/area6_data_label.txt --visu
