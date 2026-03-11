set -x

#folders=(f0 f1 f2 f3 f4)

run_dir=outputs/fundus_oct_20210825

python tools/train_net.py \
    input_type=oct \
    log_period=1 \
    data=fundus_grade_oct \
    data.train_mode=all \
    data.batch_size=32 \
    data.oct_depth=16 \
    data.oct_depth_resize=sample \
    data.number_workers=4 \
    loss=ce \
    model=resnet3d50 \
    model.pretrained=True \
    optim=sgd \
    optim.lr=0.001 \
    scheduler=step \
    scheduler.step_size=50 \
    train.max_epoch=200 \
    train.keep_checkpoint_num=1 \
    test.augment.oct.method=resize_smallest \
    test.augment.oct.resize_small_size=224 \
    hydra.run.dir=${run_dir} \
    train.resume=True

#run_dir=outputs/fundus_oct_20210815_ce_adam
#
#python tools/train_net.py \
#    input_type=oct \
#    log_period=5 \
#    data=fundus_grade_oct \
#    data.train_mode=all \
#    data.batch_size=8 \
#    data.oct_depth=16 \
#    data.oct_depth_resize=sample \
#    data.number_workers=4 \
#    loss=ce \
#    model=resnet3d50 \
#    optim=adam \
#    optim.lr=0.0001 \
#    scheduler=step \
#    scheduler.step_size=50 \
#    train.max_epoch=150 \
#    train.keep_checkpoint_num=5 \
#    test.augment.oct.method=resize_smallest \
#    test.augment.oct.resize_small_size=224 \
#    hydra.run.dir=${run_dir}

#for i in "${folders[@]}"
#do
#    python tools/train_net.py \
#        input_type=oct \
#        log_period=5 \
#        data=fundus_grade_oct_folders \
#        data.train_split=train-${i} \
#        data.val_split=val-${i} \
#        data.test_split=val-${i} \
#        data.train_mode=train \
#        data.batch_size=8 \
#        data.oct_depth=16 \
#        data.oct_depth_resize=sample \
#        data.train_mode=train \
#        data.number_workers=4 \
#        loss=ce \
#        model=resnet3d50 \
#        optim=sgd \
#        optim.lr=0.001 \
#        scheduler=step \
#        scheduler.step_size=50 \
#        train.max_epoch=250 \
#        test.augment.oct.method=resize_smallest \
#        test.augment.oct.resize_small_size=224 \
#        hydra.run.dir=${run_dir}
#done

