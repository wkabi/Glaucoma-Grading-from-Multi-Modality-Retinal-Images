set -x

run_dir=outputs/fundus_twostream_resnet34_sam_20210828

#python tools/train_net.py \
#    task=twostream \
#    input_type=multimodal \
#    data=fundus_grade_oct \
#    data.batch_size=16 \
#    data.oct_depth=16 \
#    data.oct_depth_resize=sample \
#    data.test_oct_depth_resize=sample \
#    data.oct_sample_step=4 \
#    data.train_mode=train \
#    data.val_mode=val \
#    log_period=1 \
#    loss=twostream_loss \
#    loss.weight='[1.0, 5.0]' \
#    loss.alpha=0.01 \
#    loss.max_alpha=10.0 \
#    loss.temp=1.0 \
#    loss.step_size=50 \
#    loss.step_factor=10 \
#    model=twostream_model \
#    optim=sgd optim.lr=0.01 \
#    scheduler=step scheduler.step_size=80 \
#    train.max_epoch=200 \
#    train.keep_checkpoint_num=1 \
#    test.augment.img.method=resize_smallest \
#    test.augment.img.resize_small_size='[224, 256, 384, 480]' \
#    test.augment.oct.method=resize_smallest \
#    test.augment.oct.resize_small_size='[224, 256, 384, 480]' \
#    test.augment.reduce=tsharpen \
#    hydra.run.dir=${run_dir}

python tools/train_net.py \
    task=twostream \
    input_type=multimodal \
    log_period=2 \
    data=fundus_grade_oct \
    data.train_mode=train \
    data.batch_size=16 \
    data.oct_depth_resize=sample \
    data.oct_depth=16 \
    data.number_workers=4 \
    model=twostream_resnet34 \
    optim=sam optim.lr=0.001 \
    loss=twostream_loss \
    loss.weight="[1.0, 1.0]" loss.alpha=1.0 loss.max_alpha=10.0 loss.temp=10.0 \
    loss.step_size=50 loss.step_factor=1.0 \
    scheduler=step scheduler.step_size=60 \
    train.max_epoch=200 \
    test.augment.img.method=resize_smallest \
    test.augment.img.resize_small_size="[224, 256, 384, 480]" \
    test.augment.oct.method=resize_smallest \
    test.augment.oct.resize_small_size="[224, 256, 384, 480]" \
    test.augment.reduce=tsharpen \
    hydra.run.dir=${run_dir}
