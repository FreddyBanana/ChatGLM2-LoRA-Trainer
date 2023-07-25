python inference.py \
	--CUTOFF_LEN 256 \
	--MODEL_NAME THUDM/chatglm2-6b \
	--LORA_CHECKPOINT_DIR ./output_model/model_final \
	--BIT_4 \
	--PROMPT hello
