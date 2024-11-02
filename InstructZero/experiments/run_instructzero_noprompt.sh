export CUDA_VISIBLE_DEVICES=0
SFT=5
RANDOM_PROJ='uniform'
INTRINSIC_DIM=10
model_dir='lmsys/vicuna-13b-v1.3'
MODEL_NAME='vicuna'
# model_dir='WizardLM/WizardLM-13B-V1.1'
#model_dir='WizardLMTeam/WizardLM-13B-V1.2'
#MODEL_NAME='wizardlm'

export TRANSFORMERS_CACHE=/work/dagarwal_umass_edu/HF_HOME/hub

DATASETS=(informal_to_formal odd_one_out second_word_letter synonyms word_sorting letters_list)

OUT_FILE='noprompt-temp1'

TEMPERATURE=1.0
BBOX_MODEL='gpt-3.5-turbo'
SEED=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        # Script arguments
        --bbox_model) BBOX_MODEL="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --datasets) DATASETS="$2"; shift ;;
        --whitebox_model) MODEL_NAME="$2"; shift ;;
        --temperature) TEMPERATURE="$2"; shift ;;
        *) echo "Invalid option: $1" >&2; exit 1 ;;
    esac
    shift
done

# Set model_dir based on MODEL_NAME
if [ "$MODEL_NAME" == "vicuna" ]; then
    model_dir='lmsys/vicuna-13b-v1.3'
elif [ "$MODEL_NAME" == "wizardlm" ]; then
    model_dir='WizardLMTeam/WizardLM-13B-V1.2'
fi

for i in ${DATASETS[@]}; do
    echo "### STARTING RUN FOR $i ###"  
    python InstructZero/experiments/run_instructzero.py \
    --task $i \
    --random_proj ${RANDOM_PROJ} \
    --n_prompt_tokens $SFT \
    --intrinsic_dim $INTRINSIC_DIM \
    --HF_cache_dir ${model_dir} \
    --seed ${SEED} \
    --model_name ${MODEL_NAME} \
    --bbox_model ${BBOX_MODEL} \
    --no_prompt \
    --do_sample \
    --temperature ${TEMPERATURE} \
    --out_file ${OUT_FILE}
done
