
#python batch_inference.py --model_path log6/model.ckpt --dump_dir log6/dump_eva --output_filelist log6/output_filelist_eva.txt --room_data_filelist meta/area6_data_label_1.txt --visu

#python batch_inference.py --model_path log6/model.ckpt --dump_dir log6/dump_eva_spl --output_filelist log6/output_filelist_eva_spl.txt --room_data_filelist meta/area6_data_label_split.txt --visu

#python batch_inference.py --model_path log6/model.ckpt --dump_dir log6/split_dump --output_filelist log6/split_output_filelist.txt --room_data_filelist meta/area6_data_split.txt --visu

python batch_inference.py --model_path log6/model.ckpt --dump_dir log6/dump --output_filelist log6/output_filelist.txt --room_data_filelist meta/area6_data.txt --visu
