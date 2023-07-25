from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import transformers
from datasets import load_dataset
import torch


def get_model_and_tokenizer(args, config):
    if args.BIT_8:
        model = AutoModel.from_pretrained(
            args.MODEL_NAME,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
    elif args.BIT_4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModel.from_pretrained(
            args.MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModel.from_pretrained(
            args.MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.MODEL_NAME, trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)
    model.config.use_cache = False

    return model, tokenizer


def chatglm2_tokenizer(args, tokenizer, prompt):
    data_slice = tokenizer.encode_plus(
        prompt,
        max_length=args.CUTOFF_LEN - 1,
        padding="max_length",
        truncation=True
    )
    data_slice['input_ids'].extend([tokenizer.eos_token_id])
    data_slice['attention_mask'].extend([1])
    data_slice['position_ids'].extend([data_slice['position_ids'][-1] + 1])

    return data_slice


def process_data(args, tokenizer, dataset):
    if args.MODEL_NAME == "THUDM/chatglm2-6b":
        data = dataset.shuffle().map(
            lambda data_point: chatglm2_tokenizer(
                args,
                tokenizer,
                generate_prompt(args, data_point)
            )
        )
    else:
        data = dataset.shuffle().map(
            lambda data_point: tokenizer(
                generate_prompt(args, data_point),
                truncation=True,
                max_length=args.CUTOFF_LEN,
                padding="max_length",
            )
        )

    return data


def get_lora_config(args):
    config = LoraConfig(
        r=args.LORA_R,
        lora_alpha=args.LORA_ALPHA,
        lora_dropout=args.LORA_DROPOUT,
        bias='none',
        task_type='CAUSAL_LM',
    )

    return config


def get_trainer(args, model, data, tokenizer):
    GRADIENT_ACCUMULATION_STEPS = args.BATCH_SIZE // args.MICRO_BATCH_SIZE
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data['train'],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=args.WARMUP_STEPS,
            num_train_epochs=args.EPOCHS,
            learning_rate=args.LEARNING_RATE,
            logging_steps=args.LOGGING_STEPS,
            save_strategy="steps",
            save_steps=args.SAVE_STEPS,
            output_dir=args.OUTPUT_DIR,
            overwrite_output_dir=True,
            save_total_limit=args.SAVE_TOTAL_LIMIT,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    return trainer


def get_dataset(args):
    if args.DATA_TYPE == "json":
        dataset = load_dataset("json", data_files=args.DATA_PATH)
    elif args.DATA_TYPE == "txt":
        dataset = load_dataset("text", data_files=args.DATA_PATH)

    return dataset


def generate_prompt(args, data_point):
    if args.DATA_TYPE == "json":
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: 
{data_point["context"]}

### Answer: 
{data_point["target"]}"""

    elif args.DATA_TYPE == "txt":
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### 
{data_point["text"]}"""
