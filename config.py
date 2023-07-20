import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Config of ChatGLM2-lora")

    parser.add_argument("--MICRO_BATCH_SIZE", type=int, default=4, help="Per device train batch size")
    parser.add_argument("--BATCH_SIZE", type=int, default=8, help="batch size")
    parser.add_argument('--EPOCHS', type=int, default=100, help='Training epochs')
    parser.add_argument('--WARMUP_STEPS', type=int, default=100, help='Warmup steps')
    parser.add_argument('--LEARNING_RATE', type=float, default= 2e-5 , help='Training learning rate')
    parser.add_argument('--CUTOFF_LEN', type=int, default=256, help='Truncation length')
    parser.add_argument('--LORA_R', type=int, default=16, help='Lora low rank')
    parser.add_argument('--LORA_ALPHA', type=int, default=16, help='Lora Alpha')
    parser.add_argument('--LORA_DROPOUT', type=float, default=0.05, help='Lora dropout')
    parser.add_argument('--MODEL_NAME', type=str, default="THUDM/chatglm2-6b", help='Model name')
    parser.add_argument('--LOGGING_STEPS', type=int, default=100, help='Logging steps in training')
    parser.add_argument('--OUTPUT_DIR', type=str, default="./output_model", help='Output dir')
    parser.add_argument('--DATA_PATH', type=str, default="./train_data.json", help='Input dir')
    parser.add_argument('--DATA_TYPE', type=str, choices= ["json" , "txt"], default="json", help='Input file type')
    parser.add_argument('--SAVE_STEPS', type=int, default=1000, help='Save the model according to steps')
    parser.add_argument('--SAVE_TOTAL_LIMIT', type=int, default=3, help='The number of the checkpoint you will save (Excluding the final one)')

    parser.add_argument('--PROMPT', type=str, default="Input your prompt", help='Your prompt when inference')
    parser.add_argument('--TEMPERATURE', type=int, default=0, help='Temperature when inference')
    parser.add_argument('--LORA_CHECKPOINT_DIR', type=str, default="./output_model/checkpoint-1000/", help='Your Lora checkpoint')

    return parser.parse_args()