from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import sqlite3

MODEL_NAME_OR_PATH = "llm-jp/llm-jp-3-980m-instruct2"

conn = sqlite3.connect('chat_history.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT,
    bot_response TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

def save_to_db(user_input, bot_response):
    cursor.execute("INSERT INTO chat_history(user_input, bot_response) VALUES (?, ?)", (user_input, bot_response))
    conn.commit()

def display_chat_history():
    cursor.execute("SELECT * FROM chat_history ORDER BY timestamp")
    rows = cursor.fetchall()
    for row in rows:
        print(f"{row[3]} - User: {row[1]} | Bot: {row[2]}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        device_map="auto",
    )
    model.eval()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful, concise, and accurate AI assistant.",
        },
    ]

    
    print("Enter your prompt (Ctrl-C or Ctrl-D to exit):")
    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        prompt_text = tokenizer.apply_chat_template(
            messages + [{"role": "user", "content": user_input}],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1,
        )
        response = tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        print(response)


if __name__ == "__main__":
    main()
