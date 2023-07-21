from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModel
import transformers
from datasets import load_dataset


def get_model_and_tokenizer(model_name, config):
    model = AutoModel.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, add_eos_token=True
    )
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, config)
    model.config.use_cache = False

    return model, tokenizer


def process_data(args, tokenizer, dataset):
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

