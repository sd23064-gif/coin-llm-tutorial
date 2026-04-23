from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL_NAME_OR_PATH = "llm-jp/llm-jp-3-980m-instruct2"

def main():
    print("Hello, World!")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
    )
    
    encode = tokenizer.encode("「吾輩は猫である。名前はまだない。」")
    decode = tokenizer.decode(encode)
    print("Encode:", encode)
    print("Decode:", decode)

if __name__ == "__main__":
    main()
