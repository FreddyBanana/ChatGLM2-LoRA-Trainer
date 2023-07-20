from config import parse_args
from utils import get_lora_config, get_model_and_tokenizer, process_data, get_trainer, get_dataset

if __name__ == "__main__":
    args = parse_args()

    lora_config = get_lora_config(args)
    model, tokenizer = get_model_and_tokenizer(args.MODEL_NAME, lora_config)

    dataset = get_dataset(args)
    data = process_data(args, tokenizer, dataset)

    trainer = get_trainer(args, model, data, tokenizer)
    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(args.OUTPUT_DIR + "/model_final")
