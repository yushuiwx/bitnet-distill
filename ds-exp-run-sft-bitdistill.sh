export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
export WANDB_PROJECT=<YOUR_WANDB_PROJECT>

LR=$1
MODEL=$2
GPU=$3
TEACHER_PATH=$4
BETA=$5
MINILM_WEIGHT=$6
LAYER=$7
YAML="$(pwd)/yamls/training_args/sft_nli_bitnet_allnorm_distill.yaml"
RUN_NAME="bitdistill-medical-o1-reasoning-en-${MODEL}-warm0.05-epoch6-lr${LR}-beta${BETA}-miniLM${MINILM_WEIGHT}-layer${LAYER}-bs32xa1xg1-mi300"
OUTPUT_ROOT_DIR=<YOUR_OUTPUT_ROOT_DIR>
OUTPUT_DIR="${OUTPUT_ROOT_DIR}/${RUN_NAME}"
mkdir -p ${OUTPUT_DIR}

HIP_VISIBLE_DEVICES=${GPU} llamafactory-cli train ${YAML} per_device_train_batch_size=8 \
      run_name=${RUN_NAME} \
      output_dir=${OUTPUT_DIR} \
      model_name_or_path=deepseek-ai/${MODEL} \
      gradient_accumulation_steps=1 \
      learning_rate=${LR} \
      num_train_epochs=6.0 \
      save_steps=1000 eval_steps=200 \
      per_device_eval_batch_size=8 \
      template=deepseekr1 \
      cutoff_len=4096 \
      lr_scheduler_type=cosine \
      dataset_dir=<YOUR_DATASET_DIR> \
      dataset=<YOUR_DATASET> eval_dataset=<YOUR_EVAL_DATASET> \
      weight_quant_method=minmax \
      teacher_model_path=${TEACHER_PATH} \
      beta=${BETA} \
      minilmv2_loss_coeff=${MINILM_WEIGHT} \
      layers_used_for_distill="[${LAYER}]" \
      distill_to_student_final_layer=False \
      2>&1 | tee $OUTPUT_DIR/train.log