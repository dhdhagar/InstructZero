#!/bin/bash -e

eval "$(conda shell.bash hook)"
conda deactivate
conda activate InstructZero
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/work/dagarwal_umass_edu/HF_HOME

task_dirs="InstructZero/experiments/data/semantle/train InstructZero/experiments/data/semantle/test"

for task_dir in ${task_dirs[@]}; do
  # Iterate over each csv file in the task directory
  for csv_file in $task_dir/*.csv; do
    # Get the task name from the csv file
    task_name=$(basename $csv_file .csv)
    echo "Running task: $task_name"

    # Run the task with do_sample and with no-do_sample
    for sampling in "do_sample" "no-do_sample"; do
      echo "Running with: --$sampling"

      # Run for each seed
      for seed in {42..46}; do
        echo "Running seed: $seed"

        # Run for each of random_prompt, no_prompt,
        # scores coupled_kernel, instruct-embed coupled_kernel, none coupled_kernel

        # Random prompt
        echo "Running with: random prompt"
        python InstructZero/experiments/run_semantle.py \
          --warmstart_path=$task_dir \
          --task=$task_name \
          --out_file="rand-$sampling" \
          --coupled_kernel="none" \
          --random_prompt \
          --$sampling \
          --model_name="meta-llama/Llama-3.1-8B-Instruct" \
          --bbox_model="simcse" \
          --bbox_cache="none" \
          --n_init=10 \
          --seed=$seed \
          --batch_size=1 \
          --n_iterations=200 \
          --hf_access_token="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" \
          --repetition_penalty=1.1 \
          --guesses_per_prompt=5 \
          --standardize_outputs \
          --normalize_inputs

        # No prompt (only when do_sample is enabled)
        if [ $sampling == "do_sample" ]; then
          echo "Running with: no prompt"
          python InstructZero/experiments/run_semantle.py \
            --warmstart_path=$task_dir \
            --task=$task_name \
            --out_file="repeated-sampling" \
            --coupled_kernel="none" \
            --no_prompt \
            --$sampling \
            --model_name="meta-llama/Llama-3.1-8B-Instruct" \
            --bbox_model="simcse" \
            --bbox_cache="none" \
            --n_init=10 \
            --seed=$seed \
            --batch_size=1 \
            --n_iterations=200 \
            --hf_access_token="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" \
            --repetition_penalty=1.1 \
            --guesses_per_prompt=5 \
            --standardize_outputs \
            --normalize_inputs
        fi

        for coupled_kernel in "scores" "instruct-embed" "none"; do
          echo "Running with: $coupled_kernel coupled kernel"

          python InstructZero/experiments/run_semantle.py \
            --warmstart_path=$task_dir \
            --task=$task_name \
            --out_file="coupled-$coupled_kernel-$sampling" \
            --coupled_kernel=$coupled_kernel \
            --$sampling \
            --model_name="meta-llama/Llama-3.1-8B-Instruct" \
            --bbox_model="simcse" \
            --bbox_cache="none" \
            --n_init=10 \
            --seed=$seed \
            --batch_size=1 \
            --n_iterations=200 \
            --hf_access_token="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" \
            --repetition_penalty=1.1 \
            --guesses_per_prompt=5 \
            --standardize_outputs \
            --normalize_inputs
        done
      done
    done
  done
done
