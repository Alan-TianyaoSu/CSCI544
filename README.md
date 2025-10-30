# CSCI544
Identifying Human Preferred Chatbot Responses

## Requirements
- Python == 3.12

## Model List
- QWEN2.5-1.5B
- QWEN2.5-3B
- GEMMA3-1B
- GEMMA3-4B
- Adding more...

## How to use
- Put .parquet in ./dataset_raw (train, val, test)
- Use ./data_process/data_split.py to get numpy data in ./dataset
- Add all the model you want to use in ./model/config.py _DEFAULT_MODELS
- Use model_download.py to download all the models in _DEFAULT_MODELS
- All the training scripts locate in ./training, please read the "To use" part on the top
- All the parameters should be writen in ./training files
- Use ./evaluate/select_winner.py to check training result

## Status
- Week 1 — in progress  
  - run text cleaning, deduplication, language validation, quality filtering  
  - build SFT single-sided and DPO paired views for train/val/test  
  - implement no-training baseline evaluation pipeline  
  - record pairwise accuracy, likelihood gap, per-language metrics

- Check List:
  - Basic Data Analysis (Remain improving)
  - Data Processing (To Do)
  - SFT Pipeline (In Progress)

## Remaining
- Week 2 — pending  
  - complete SFT pipeline with QLoRA  
  - train on single-sided pairs with weighted low-resource/code data  
  - tune micro-batch, gradient accumulation, max sequence length  
  - run pilots, full SFT training, early stopping vs baseline

- Week 3 — pending  
  - sweep SFT hyperparameters (lr, LoRA rank, alpha, dropout)  
  - retrain with best settings, save top checkpoint  
  - implement DPO/IPO pipeline from best SFT checkpoint  
  - apply stratified sampling, KL regularization, reduced lr, gradient clipping

- Week 4 — pending  
  - tune DPO hyperparameters (β/λ, lr, regularization)  
  - compute accuracy, AUROC, likelihood margin, per-language metrics  
  - run robustness checks, statistical tests  
  - finalize report, visuals, ablations, presentation, reproducibility docs

