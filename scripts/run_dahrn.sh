dataroot="../datasets/SYSU-CD/train/"
val_dataroot="../datasets/SYSU-CD/val/"
test_dataroot="../datasets/SYSU-CD/test/"
lr=0.0001
model=DAHRN
batch_size=16
num_threads=4
save_epoch_freq=1
angle=20
gpu=1
port=8091
preprocess='blur_rotate_transpose_hsvshift_noise_flip'
name='dahrn'
criterion='hybrid_bcl'
# python ./train.py --epoch_count 1  -l $criterion --preprocess $preprocess  --display_port $port --gpu_ids $gpu --num_threads $num_threads --save_epoch_freq $save_epoch_freq --angle $angle  --dataroot ${dataroot} --val_dataroot ${val_dataroot} --test_dataroot ${test_dataroot}  --name $name --lr $lr --model $model  --batch_size $batch_size 

dataroot="../datasets/SYSU-CD/train/"
val_dataroot="../datasets/SYSU-CD/val/"
test_dataroot="../datasets/SYSU-CD/test/"
lr=0.0001
model=DAHRN
batch_size=16
num_threads=4
save_epoch_freq=1
angle=20
gpu=1
port=8091
preprocess='blur_rotate_transpose_hsvshift_noise_flip'
name='SYSU_DAHRN_1210'
criterion='hybrid_bcl'
python ./train.py --epoch_count 1  -l $criterion --preprocess $preprocess  --display_port $port --gpu_ids $gpu --num_threads $num_threads --save_epoch_freq $save_epoch_freq --angle $angle  --dataroot ${dataroot} --val_dataroot ${val_dataroot} --test_dataroot ${test_dataroot}  --name $name --lr $lr --model $model  --batch_size $batch_size 
