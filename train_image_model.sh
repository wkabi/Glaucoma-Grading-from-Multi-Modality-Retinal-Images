set -x

run_dir=outputs/fundus_img_20210817_ce_sgd

python tools/train_net.py \
    data=fundus_grade_img \
    data.data_root="/home/bliu/work/Code/Optic/data/GAMMA_training_data/training_data" \
    data.batch_size=64 \
    data.train_mode=train \
    data.val_mode=val \
    input_type=image \
    log_period=1 \
    loss=ce \
    model=resnet50 \
    optim=sgd optim.lr=0.01 \
    scheduler=step scheduler.step_size=40 \
    train.max_epoch=100 \
    train.keep_checkpoint_num=5 \
    test.augment.img.method=resize_smallest \
    test.augment.img.resize_small_size=224 \
    wandb.enable=False \
    hydra.run.dir=${run_dir}

#run_dir=outputs/fundus_img_20210816_ce_adam
#
#python tools/train_net.py \
#    data=fundus_grade_img \
#    data.batch_size=64 \
#    data.train_mode=all \
#    input_type=image \
#    log_period=1 \
#    loss=ce \
#    model=resnet50 \
#    optim=adam optim.lr=0.001 \
#    scheduler=step scheduler.step_size=40 \
#    train.max_epoch=100 \
#    train.keep_checkpoint_num=5 \
#    test.augment.img.method=resize_smallest \
#    test.augment.img.resize_small_size=224 \
#    hydra.run.dir=${run_dir}
#folders=(f0 f1 f2 f3 f4)
#folders=(f0 f1 f2 f3 f4)

#run_dir=outputs/fundus_img_20210815
#
#for i in "${folders[@]}"
#do
#    python tools/train_net.py \
#        data=fundus_grade_img_folders \
#        data.batch_size=64 \
#        data.train_split=train-${i} \
#        data.val_split=val-${i} \
#        data.test_split=val-${i} \
#        data.train_mode=train \
#        input_type=image \
#        log_period=1 \
#        loss=ce \
#        model=resnet50 \
#        optim=sgd optim.lr=0.01 \
#        scheduler=step scheduler.step_size=80 \
#        train.max_epoch=200 \
#        test.augment.img.method=resize_smallest \
#        test.augment.img.resize_small_size=224 \
#        hydra.run.dir=${run_dir}
#    #echo $i
#done



