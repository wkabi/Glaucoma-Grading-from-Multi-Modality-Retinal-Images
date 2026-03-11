set -x

#data_root=/home/bliu/work/Code/Optic/data/GAMMA_training_data/training_data
test_mode=val
#data_root=/home/ar88770/Optic-main/training_data
data_root=/home/wkabir/Project_Multimodal/Multimodal_Glaucoma/Optic-main/training_data
#test_mode=all

#run_dir=outputs/fundus_twostream_resnet34_sam_20210826
#run_dir=outputs/fundus_multimodal_resnet34-50_sam_20210830
run_dir=outputs/fundus_multimodal_sam_20210830

#checkpoint=jagusifd-twostream_model-twostream_loss-train-best

#MODELS=(\
#    3qvibp5v-multimodal-multimodal_ce_kl-train-f0-best \
#)
    #3n9pkth4-multimodal-multimodal_ce_kl-train-f0-best
    #2ejeft5b-multimodal-multimodal_ce_kl-train-f2-best \
    #34pqqgrg-multimodal-multimodal_ce_kl-train-f1-best \

#for m in "${MODELS[@]}"
#do
python tools/test_net.py \
    task=multimodal \
    input_type=multimodal \
    data=fundus_grade_img_oct \
    data.data_root=${data_root} \
    data.test_mode=${test_mode} \
    data.oct_depth=16 \
    data.oct_depth_resize=sample \
    +data.test_oct_depth_resize=sample \
    +data.oct_sample_step=4 \
    log_period=5 \
    model=multimodal \
    model.oct_model_name=resnet34 \
    model.oct_pretrained=False \
    model.img_model_name=resnet34 \
    model.oct_shortcut_type=A \
    test.augment.img.method=resize_smallest \
    test.augment.img.resize_small_size='[224, 256, 384, 480]' \
    test.augment.img.flip=False \
    test.augment.oct.method=resize_smallest \
    test.augment.oct.resize_small_size='[224, 256, 384, 480]' \
    test.augment.oct.flip=False \
    test.augment.reduce=tsharpen \
    hydra.run.dir=${run_dir} \
    test.checkpoint=best.pth \
    test.save_prediction=${test_mode}.txt

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



