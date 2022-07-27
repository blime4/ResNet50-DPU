

MODEL_DIR="./tmp-726-batch_size_256-dtype=fp32_FILES=512-DPU-again"
DATASET_DIR="/LocalRun/shaobo.xie/datasets/resnet/ILSVRC2012/tf_records/train"
# export CUDA_VISIBLE_DEVICES=1

#--model_dir=$MODEL_DIR \
# --noskip_eval \

python3 ./resnet_ctl_imagenet_main.py \
--base_learning_rate=8.5 \
--batch_size=256 \
--enable_tensorboard \
--data_dir=$DATASET_DIR \
--model_dir=$MODEL_DIR \
--datasets_num_private_threads=32 \
--dtype=fp32 \
--device_warmup_steps=1 \
--noenable_device_warmup \
--enable_eager \
--noenable_xla \
--epochs_between_evals=4 \
--noeval_dataset_cache \
--eval_offset_epochs=2 \
--eval_prefetch_batchs=192 \
--label_smoothing=0.1 \
--log_steps=1 \
--lr_schedule=polynomial \
--momentum=0.9 \
--num_accumulation_steps=1 \
--num_classes=1000 \
--num_gpus=2 \
--lars_epsilon=0 \
--optimizer=LARS \
--report_accuracy_metrics \
--steps_per_loop=1252 \
--target_accuracy=0.759 \
--tf_data_experimental_slack \
--tf_gpu_thread_mode=gpu_private \
--notrace_warmup \
--train_epochs=41 \
--training_prefetch_batchs=128 \
--nouse_synthetic_data \
--warmup_epochs=1 \
--weight_decay=0.0002 \
--enable_checkpoint_and_export \
--clean \
--explicit_gpu_placement

