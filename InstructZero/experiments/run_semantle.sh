export CUDA_VISIBLE_DEVICES="0,1"
SFT=5
RANDOM_PROJ='uniform'
INTRINSIC_DIM=10
# model_dir='WizardLMTeam/WizardLM-13B-V1.2'
# MODEL_NAME='wizardlm'
model_dir="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME='llama3-8B'
API_MODEL="chatgpt"
export HF_HOME=/scratch/workspace/vpimpalkhute_umass_edu-bo_llm/

datasets=(cement child computer crane meatloaf papa polyethylene sax trees ) # birthstone
# informal_to_formal odd_one_out second_word_letter synonyms word_sorting letters_list)

for i in ${datasets[@]}; do
    echo $i
    python experiments/run_semantle.py \
    --task $i \
    --random_proj ${RANDOM_PROJ} \
    --n_prompt_tokens $SFT \
    --intrinsic_dim $INTRINSIC_DIM \
    --HF_cache_dir ${model_dir} \
    --model_name ${MODEL_NAME} \
    --api_model ${API_MODEL} \
    --instruct_method vec_sim \
    --semantle_word $i \
    # --score_method task_similarity 
done
