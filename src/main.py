from data_preparation import prepare_data
from config.config import device
from utils import train_model, generate_text
from models.model import GPTLanguageModel

def main():
    # Prepare the data
    train_data, val_data, stoi, itos, vocab_size = prepare_data()

    # Define the GPTLanguageModel
    model = GPTLanguageModel(vocab_size=vocab_size)
    model.to(device)

    # Train the model
    train_model(model, train_data, val_data)

    # Context for text generation
    context = "Once upon a time"

    # Number of new tokens to generate
    max_new_tokens = 100

    # Generate text using the trained model
    generated_text = generate_text(model, context, max_new_tokens, stoi, itos)

    # Print the generated text
    print("Generated text:")
    print(generated_text)


if __name__ == "__main__":
    main()
