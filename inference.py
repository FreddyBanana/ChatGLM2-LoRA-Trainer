from config import parse_args
from transformers import AutoModel
from transformers import AutoTokenizer
from peft import PeftModel
import torch


if __name__ == "__main__":
    args = parse_args()
    model = AutoModel.from_pretrained(args.MODEL_NAME, trust_remote_code=True, load_in_8bit=False,
                                      device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.MODEL_NAME, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.LORA_CHECKPOINT_DIR)

    with torch.no_grad():
        ids = tokenizer.encode(args.PROMPT)
        input_ids = torch.LongTensor([ids])
        out = model.generate(
            input_ids=input_ids,
            max_length=args.CUTOFF_LEN,
            do_sample=False,
            temperature=args.TEMPERATURE
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(args.PROMPT, "").replace("\nEND", "").strip()
        print(f"### .Answer:\n", answer, '\n\n')