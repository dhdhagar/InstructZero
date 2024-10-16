export CUDA_VISIBLE_DEVICES=0
SFT=5
RANDOM_PROJ='uniform'
INTRINSIC_DIM=10
# model_dir='lmsys/vicuna-13b-v1.3'
# MODEL_NAME='vicuna'
# model_dir='WizardLM/WizardLM-13B-V1.1'
model_dir='WizardLMTeam/WizardLM-13B-V1.2'
MODEL_NAME='wizardlm'
export TRANSFORMERS_CACHE=./transformers_cache

datasets=(informal_to_formal odd_one_out second_word_letter synonyms word_sorting letters_list)

OUT_FILE='temp1'
BBOX_MODEL='gpt-4-turbo'

for i in ${datasets[@]}; do
    echo $i
    python InstructZero/experiments/run_instructzero.py \
    --task $i \
    --random_proj ${RANDOM_PROJ} \
    --n_prompt_tokens $SFT \
    --intrinsic_dim $INTRINSIC_DIM \
    --HF_cache_dir ${model_dir} \
    --model_name ${MODEL_NAME} \
    --out_file ${OUT_FILE} \
    --do_sample \
    --bbox_model ${BBOX_MODEL}
done
