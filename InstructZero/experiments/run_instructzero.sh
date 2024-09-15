export CUDA_VISIBLE_DEVICES="0,1"
SFT=5
RANDOM_PROJ='uniform'
INTRINSIC_DIM=10
# model_dir='lmsys/vicuna-13b-v1.3'
# MODEL_NAME='vicuna'
# model_dir='google/gemma-1.1-2b-it'
# MODEL_NAME='gemma'
# model_dir="meta-llama/Meta-Llama-3-8B"
# MODEL_NAME='llama3-8B'
model_dir='WizardLMTeam/WizardLM-13B-V1.2'
MODEL_NAME='wizardlm'
# API_MODEL='llama3-8b'
API_MODEL="chatgpt"
export HF_HOME=/scratch/workspace/vpimpalkhute_umass_edu-bo_llm/

export OPENAI_API_KEY=""
datasets=(word_sorting)
# informal_to_formal odd_one_out second_word_letter synonyms word_sorting letters_list)

for i in ${datasets[@]}; do
    echo $i
    python experiments/run_instructzero.py \
    --task $i \
    --random_proj ${RANDOM_PROJ} \
    --n_prompt_tokens $SFT \
    --intrinsic_dim $INTRINSIC_DIM \
    --HF_cache_dir ${model_dir} \
    --model_name ${MODEL_NAME} \
    --api_model ${API_MODEL} \
    # --score_method task_similarity 
done
