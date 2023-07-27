python main.py \
	--MICRO_BATCH_SIZE 8 \
	--BATCH_SIZE 16 \
	--EPOCHS 50 \
	--LEARNING_RATE 2e-5 \
	--CONTEXT_LEN 64 \
	--TARGET_LEN 192 \
	--LORA_R 16 \
	--MODEL_NAME THUDM/chatglm2-6b \
	--OUTPUT_DIR ./output_model \
	--DATA_PATH ./new_train.json \
	--DATA_TYPE json \
	--SAVE_STEPS 1000 \
	--BIT_4
