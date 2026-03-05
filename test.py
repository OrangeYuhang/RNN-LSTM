from task1 import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_path = './ptb/ptb.train.txt'
test_path = './ptb/ptb.test.txt'
valid_path = './ptb/ptb.valid.txt'

word2idx, idx2word = build_vocab(train_path, min_freq=1)
vocab_size = len(word2idx)
    
embed_size = 300
hidden_size = 300
num_layers_rnn = 2
num_layers_lstm = 2
dropout = 0.5
seq_len = 35

rnn = VanillaRNNLM(vocab_size, embed_size, hidden_size, num_layers_rnn, dropout).to(device)
lstm = LSTMLM(vocab_size, embed_size, hidden_size, num_layers_lstm, dropout).to(device)

rnn.load_state_dict(torch.load('best_rnn_model.pt'))
lstm.load_state_dict(torch.load('best_lstm_model.pt'))

print("\n" + "="*50)
print("Text Generation Examples")
print("="*50)
    
prompts = ["the company", "the stock market", "the president said"]
    
print("\nVanilla RNN Generated Text:")
for prompt in prompts:
    generated = generate_text(rnn, prompt, word2idx, idx2word, max_len=50)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}\n")

    
print("\nLSTM Generated Text:")
for prompt in prompts:
    generated = generate_text(lstm, prompt, word2idx, idx2word, max_len=50)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}\n") 
    

