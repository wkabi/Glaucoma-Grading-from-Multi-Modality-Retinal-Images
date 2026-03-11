#set -x

save_dir=outputs/submit

data_root=/home/bliu/work/Code/Optic/data/GAMMA_training_data/val_data
test_mode=all

results=""
# twostream_resnet50-50
run_dir=outputs/best_models/twostream_resnet50-50
MODELS=(\
    jagusifd-twostream_model-twostream_loss-train-best \
    1ka2gqz9-twostream_model-twostream_loss-train-f1-best \
)

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
        model.oct_pretrained=False \
        model.img_model_name=resnet50 \
        test.augment.img.method=resize_smallest \
        test.augment.img.resize_small_size='[224, 256, 384, 480, 640]' \
        test.augment.img.flip=False \
        test.augment.oct.method=resize_smallest \
        test.augment.oct.resize_small_size='[224, 256, 384, 480, 640]' \
        test.augment.oct.flip=False \
        test.augment.reduce=tsharpen \
        hydra.run.dir=${run_dir} \
        test.checkpoint=${m}.pth \
        test.save_prediction=${m}-${test_mode}.txt

    if [ -z "${results}" ]
    then
        results=${run_dir}/${m}-${test_mode}.txt
    else
        results=${results}" "${run_dir}/${m}-${test_mode}.txt
    fi
done

# multimodal_resnet34-34
run_dir=outputs/best_models/multimodal_resnet34-34
MODELS=(\
    1ia7qnp9-multimodal-multimodal_ce_kl-train-f3-best \
)

for m in "${MODELS[@]}"
do
    python tools/test_net.py \
        task=multimodal \
        input_type=multimodal \
        data=fundus_grade_oct \
        data.data_root=${data_root} \
        data.test_mode=${test_mode} \
        data.oct_depth=16 \
        data.oct_depth_resize=sample \
        data.test_oct_depth_resize=sample \
        data.oct_sample_step=4 \
        log_period=5 \
        model=multimodal \
        model.oct_model_name=resnet34 \
        model.oct_shortcut_type=A \
        model.oct_pretrained=False \
        model.img_model_name=resnet34 \
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

    if [ -z "${results}" ]
    then
        results=${run_dir}/${m}-${test_mode}.txt
    else
        results=${results}" "${run_dir}/${m}-${test_mode}.txt
    fi
done

# multimodal_resnet34-50
run_dir=outputs/best_models/multimodal_resnet34-50
MODELS=(\
    3d5vhpld-multimodal-multimodal_ce_kl-train-f0-best \
)

for m in "${MODELS[@]}"
do
    python tools/test_net.py \
        task=multimodal \
        input_type=multimodal \
        data=fundus_grade_oct \
        data.data_root=${data_root} \
        data.test_mode=${test_mode} \
        data.oct_depth=16 \
        data.oct_depth_resize=sample \
        data.test_oct_depth_resize=sample \
        data.oct_sample_step=4 \
        log_period=5 \
        model=multimodal \
        model.oct_model_name=resnet34 \
        model.oct_shortcut_type=A \
        model.oct_pretrained=False \
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

    if [ -z "${results}" ]
    then
        results=${run_dir}/${m}-${test_mode}.txt
    else
        results=${results}" "${run_dir}/${m}-${test_mode}.txt
    fi
done

#echo ${results}

python tools/ensemble.py ${results} --save_dir ${save_dir}
