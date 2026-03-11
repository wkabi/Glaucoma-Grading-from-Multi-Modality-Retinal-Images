set -x

run_dir=outputs/fundus_multimodal_sam_20210830

python tools/train_net.py \
    task=multimodal \
    input_type=multimodal \
    log_period=1 \
    data=fundus_grade_oct \
    data.batch_size=16 \
    data.oct_depth=16 \
    data.oct_depth_resize=sample \
    data.test_oct_depth_resize=sample \
    data.number_workers=4 \
    data.train_mode=train \
    data.val_mode=val \
    loss=multimodal_ce_kl \
    loss.weight='[1.0, 1.0]' \
    loss.alpha=1.0 \
    loss.max_alpha=10.0 \
    loss.temp=1.0 \
    loss.step_size=50 \
    loss.step_factor=1 \
    loss.kl_lambda=0.1 \
    model=multimodal \
    model.oct_model_name=resnet34 \
    model.oct_pretrained=False \
    model.img_model_name=resnet34 \
    optim=sam optim.lr=0.001 \
    scheduler=step scheduler.step_size=60 \
    train.max_epoch=200 \
    train.keep_checkpoint_num=1 \
    test.augment.img.method=resize_smallest \
    test.augment.img.resize_small_size='[224, 256, 384, 480]' \
    test.augment.oct.method=resize_smallest \
    test.augment.oct.resize_small_size='[224, 256, 384, 480]' \
    test.augment.reduce=tsharpen \
    hydra.run.dir=${run_dir} \
    wandb.enable=False

#model.img_model_checkpoint=/home/bliu/work/Code/Optic/outputs/trained_model/0a118x26-resnet50-weighted_ce-best.pth \
#model.oct_model_checkpoint=/home/bliu/work/Code/Optic/outputs/trained_model/1uhyx1nt-resnet3d50-ce-oct-train-best.pth \
