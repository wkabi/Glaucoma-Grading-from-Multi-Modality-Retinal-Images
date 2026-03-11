set -x

#data_root=/home/bliu/work/Code/Optic/data/GAMMA_training_data/training_data
#test_mode=val
data_root=/home/bliu/work/Code/Optic/data/GAMMA_training_data/val_data
test_mode=all

#run_dir=outputs/fundus_twostream_resnet34_sam_20210826
run_dir=outputs/fundus_twostream_resnet50_sam_20210826

#checkpoint=jagusifd-twostream_model-twostream_loss-train-best

MODELS=(\
    3ob3pynb-twostream_model-twostream_loss-train-f0-best \
)
    #mxq7abdr-twostream_model-twostream_loss-train-f0-best \
    #27lsdyzf-twostream_model-twostream_loss-train-f1-best \
    #fq6yeq2s-twostream_model-twostream_loss-train-f4-best \
    #22ip0a8h-twostream_model-twostream_loss-train-f2-best \
    #zjdre2pl-twostream_model-twostream_loss-train-f2-best \
    #1ka2gqz9-twostream_model-twostream_loss-train-f1-best \
    #3o7mg3bt-twostream_model-twostream_loss-train-f0-best \
    #26n0988d-twostream_model-twostream_loss-train-f0-best \
    #3cl82a4z-twostream_model-twostream_loss-train-f0-best \
    #26n0988d-twostream_model-twostream_loss-train-f0-best \

for m in "${MODELS[@]}"
do
    python tools/test_net.py \
        task=twostream \
        input_type=multimodal \
        data=fundus_grade_oct \
        data.data_root=${data_root} \
        data.test_mode=${test_mode} \
        data.oct_depth=16 \
        data.oct_depth_resize=sample \
        data.test_oct_depth_resize=sample \
        data.oct_sample_step=4 \
        log_period=5 \
        model=twostream_model \
        model.oct_model_name=resnet50 \
        model.img_model_name=resnet50 \
        test.augment.img.method=resize_smallest \
        test.augment.img.resize_small_size='[224, 256, 384, 480]' \
        test.augment.img.flip=False \
        test.augment.oct.method=resize_smallest \
        test.augment.oct.resize_small_size='[224, 256, 384, 480]' \
        test.augment.oct.flip=False \
        test.augment.reduce=tsharpen \
        hydra.run.dir=${run_dir} \
        test.checkpoint=${m}.pth \
        test.save_prediction=${m}-${test_mode}.txt
done

#    test.augment.img.resize_small_size='[224, 256, 384, 480, 640]' \
#    test.augment.oct.resize_small_size='[224, 256, 384, 480, 640]' \
#    test.augment.img.resize_small_size=224 \
#    test.augment.oct.resize_small_size=224 \
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



