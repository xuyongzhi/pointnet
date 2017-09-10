
python train_sorted.py  --gpu 0    --log_dir log6_4096_office  --eval_area 6 --max_epoch 1 --batch_size 4096 --num_point 4096   --all_fn_glob stanford_indoor3d_normedh5_stride_0.5_step_1_4096/*office*  --train    _data_rate 0.05 --eval_data_rate 0.05



#python train_sorted.py  --gpu 0    --log_dir log6_0d001_4096  --eval_area 6 --max_epoch 2 --batch_size 8 --num_point 4096   --all_fn_glob stanford_indoor3d_normedh5_stride_0.5_step_1_4096/*    --train_data_rate 0.001 --eval_data_rate 0.001
