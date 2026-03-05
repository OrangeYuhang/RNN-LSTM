import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import math
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


torch.manual_seed(42)
np.random.seed(42)

class PTBDataset(Dataset):
    def __init__(self, root:str, word2idx:dict,seq_len:int=35)->None:
        self.root = root
        self.word2idx = word2idx
        self.seq_len = seq_len
        try:
            with open(root) as f:
                text = f.read().split()
        except FileNotFoundError as e:
            print(f'FILE NOT FOUND: {e}')

        self.data = [word2idx.get(word, word2idx['<unk>']) for word in text]

        self.data.append(word2idx['<eos>'])

        self.num_batches = len(self.data)//self.seq_len

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if idx >= self.num_batches:
            raise IndexError

        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len

        x = torch.tensor(self.data[start_idx:end_idx], dtype=torch.long)
        y = torch.tensor(self.data[start_idx+ 1:end_idx+ 1], dtype=torch.long)

        return x, y

class VanillaRNNLM(nn.Module):

    def __init__(self, vocab_size:int, embed_size:int,hidden_size:int, num_layers:int=3, dropout:float =0.5)->None:
        super(VanillaRNNLM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed_size

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)

        embeds = self.embedding(x)
        embeds = self.dropout(embeds)

        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        output, hidden = self.rnn(embeds, hidden)

        output = self.dropout(output)

        logits = self.fc(output)

        return logits, hidden

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=3, dropout=0.5):
        super(LSTMLM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        embeds = self.embedding(x)
        embeds = self.dropout(embeds)
        
        if hidden is None:
            hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                      torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))
        
        output, hidden = self.lstm(embeds, hidden)
        
        output = self.dropout(output)
        
        logits = self.fc(output) 
        
        return logits, hidden

def build_vocab(data_path:str, min_freq:int =1)-> tuple[dict[str, int], dict[int, str]]:
    try:
        with open(data_path, 'r') as f:
            text = f.read().lower().split()
    except FileNotFoundError as e:
        print(f'FILE NOT FOUND: {e}')

    word_counts = Counter(text)

    vocab = ['<pad>', '<eos>']#数据集中已经含有<unk>，所以不添加

    for word, count in word_counts.items():
        vocab.append(word)

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer, clip: float = 0.25):
    model.train()
    total_loss = 0
    total_batches = 0
    hidden = None
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output, hidden = model(x, hidden)
        
        if isinstance(model, VanillaRNNLM):
            hidden = hidden.detach()
        else:  # LSTM
            hidden = (hidden[0].detach(), hidden[1].detach())
        
        loss = criterion(output.view(-1, model.vocab_size), y.view(-1))
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_batches += 1
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    
    return total_loss / total_batches

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.CrossEntropyLoss):
    model.eval()
    total_loss = 0
    total_batches = 0
    hidden = None
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            output, hidden = model(x, hidden)
            
            if isinstance(model, VanillaRNNLM):
                hidden = hidden.detach()
            else:
                hidden = (hidden[0].detach(), hidden[1].detach())
            
            loss = criterion(output.view(-1, model.vocab_size), y.view(-1))
            
            total_loss += loss.item()
            total_batches += 1
            
    return total_loss / total_batches

def generate_text(model: nn.Module, start_words: str, word2idx: dict[str, int], idx2word: dict[int, str], max_len: int = 50, temperature: float = 1.0):
    model.eval()
    
    words = start_words.lower().split()
    indices = [word2idx.get(word, word2idx['<unk>']) for word in words]
    
    input_tensor = torch.tensor(indices).unsqueeze(0).to(device)
    
    hidden = None
    generated = words.copy()
    
    with torch.no_grad():
        for _ in range(max_len):
            output, hidden = model(input_tensor, hidden)
            
            logits = output[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            next_idx = int(torch.multinomial(probs, 1).item())
            
            next_word = idx2word[next_idx]
            
            if next_word == '<eos>':
                break
            
            generated.append(next_word)
            
            input_tensor = torch.tensor([[next_idx]],dtype=torch.long).to(device)
    
    return ' '.join(generated)

def main():
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
    batch_size = 20
    learning_rate = 1e-3
    num_epochs = 40
    clip = 0.25
    
    train_dataset = PTBDataset(train_path,word2idx,seq_len)
    valid_dataset = PTBDataset(valid_path,word2idx,seq_len)
    test_dataset = PTBDataset(test_path,word2idx,seq_len)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True,drop_last=True)
    
    rnn_model = VanillaRNNLM(vocab_size, embed_size, hidden_size, num_layers_rnn, dropout).to(device)
    lstm_model = LSTMLM(vocab_size, embed_size, hidden_size, num_layers_lstm, dropout).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
    
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
    
    rnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(rnn_optimizer, mode='min', factor=0.5, patience=2)
    lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(lstm_optimizer, mode='min', factor=0.5, patience=2)
    
    rnn_train_losses = []
    rnn_valid_losses = []
    rnn_valid_ppls = []
    lstm_train_losses = []
    lstm_valid_losses = []
    lstm_valid_ppls = []
    
    # 训练Vanilla RNN
    print("\n" + "="*50)
    print("Training Vanilla RNN Language Model")
    print("="*50)
    
    best_rnn_ppl = float('inf')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss = train_epoch(rnn_model, train_loader, criterion, rnn_optimizer, clip)
        
        valid_loss = evaluate(rnn_model, valid_loader, criterion)
        
        train_ppl = math.exp(train_loss)
        valid_ppl = math.exp(valid_loss)
        
        rnn_train_losses.append(train_loss)
        rnn_valid_losses.append(valid_loss)
        rnn_valid_ppls.append(valid_ppl)
        
        rnn_scheduler.step(valid_loss)
        
        if valid_ppl < best_rnn_ppl:
            best_rnn_ppl = valid_ppl
            torch.save(rnn_model.state_dict(), 'best_rnn_model.pt')
            
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s | LR: {rnn_scheduler.get_last_lr()[0]:.4f}')
        print(f'Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}')
        print(f'Valid Loss: {valid_loss:.4f} | Valid PPL: {valid_ppl:.2f}')
        print(f'Best Valid PPL: {best_rnn_ppl:.2f}')
        print('-'*50)
        
    rnn_converge_epoch = None
    for i, ppl in enumerate(rnn_valid_ppls):
        if ppl <= best_rnn_ppl * 1.05:
            rnn_converge_epoch = i + 1
            break
    
    # 训练LSTM
    print("\n" + "="*50)
    print("Training LSTM Language Model")
    print("="*50)
    
    best_lstm_ppl = float('inf')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss = train_epoch(lstm_model, train_loader, criterion, lstm_optimizer, clip)
        
        valid_loss = evaluate(lstm_model, valid_loader, criterion)
        
        train_ppl = math.exp(train_loss)
        valid_ppl = math.exp(valid_loss)
        
        lstm_train_losses.append(train_loss)
        lstm_valid_losses.append(valid_loss)
        lstm_valid_ppls.append(valid_ppl)
        
        lstm_scheduler.step(valid_loss)
        
        if valid_ppl < best_lstm_ppl:
            best_lstm_ppl = valid_ppl
            torch.save(lstm_model.state_dict(), 'best_lstm_model.pt')
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s | LR: {lstm_scheduler.get_last_lr()[0]:.4f}')
        print(f'Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}')
        print(f'Valid Loss: {valid_loss:.4f} | Valid PPL: {valid_ppl:.2f}')
        print(f'Best Valid PPL: {best_lstm_ppl:.2f}')
        print('-'*50)
        
    lstm_converge_epoch = None
    for i, ppl in enumerate(lstm_valid_ppls):
        if ppl <= best_lstm_ppl * 1.05:
            lstm_converge_epoch = i + 1
            break
    
    # 测试最佳模型
    print("\n" + "="*50)
    print("Testing Best Models")
    print("="*50)
    
    rnn_model.load_state_dict(torch.load('best_rnn_model.pt'))
    lstm_model.load_state_dict(torch.load('best_lstm_model.pt'))
    
    rnn_test_loss = evaluate(rnn_model, test_loader, criterion)
    lstm_test_loss = evaluate(lstm_model, test_loader, criterion)
    
    rnn_test_ppl = math.exp(rnn_test_loss)
    lstm_test_ppl = math.exp(lstm_test_loss)
    
    print(f"Vanilla RNN - Test Loss: {rnn_test_loss:.4f} | Test PPL: {rnn_test_ppl:.2f}")
    print(f"LSTM - Test Loss: {lstm_test_loss:.4f} | Test PPL: {lstm_test_ppl:.2f}")
    
    # 绘制训练/验证 Loss 曲线
    print("\n" + "="*50)
    print("Generating Training Curves")
    print("="*50)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs_range = range(1, num_epochs + 1)
    
    axes[0].plot(epochs_range, rnn_train_losses, 'b-', label='RNN Train Loss', linewidth=2)
    axes[0].plot(epochs_range, rnn_valid_losses, 'b--', label='RNN Valid Loss', linewidth=2)
    axes[0].plot(epochs_range, lstm_train_losses, 'r-', label='LSTM Train Loss', linewidth=2)
    axes[0].plot(epochs_range, lstm_valid_losses, 'r--', label='LSTM Valid Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Train/Valid Loss Comparison: RNN vs LSTM', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs_range, rnn_valid_ppls, 'b--', label='RNN Valid PPL', linewidth=2)
    axes[1].plot(epochs_range, lstm_valid_ppls, 'r--', label='LSTM Valid PPL', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Perplexity', fontsize=12)
    axes[1].set_title('Valid Perplexity Comparison: RNN vs LSTM', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./img/training_curves.png', dpi=300, bbox_inches='tight')
    print("Saved training curves to 'training_curves.png'")
    plt.show()
    
    # 创建对比表格
    print("\n" + "="*50)
    print("Model Comparison Summary")
    print("="*50)
    
    comparison_data = {
        'Model': ['Vanilla RNN', 'LSTM'],
        'Valid PPL': [f'{best_rnn_ppl:.2f}', f'{best_lstm_ppl:.2f}'],
        'Test PPL': [f'{rnn_test_ppl:.2f}', f'{lstm_test_ppl:.2f}'],
        'Convergence Epoch': [rnn_converge_epoch, lstm_converge_epoch],
        'Best Valid Loss': [f'{rnn_valid_losses[np.argmin(rnn_valid_ppls)]:.4f}', 
                           f'{lstm_valid_losses[np.argmin(lstm_valid_ppls)]:.4f}']
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\n" + df_comparison.to_string(index=False))
    
    df_comparison.to_csv('model_comparison.csv', index=False)
    print("\nSaved comparison table to 'model_comparison.csv'")
    
    # 文本生成示例
    print("\n" + "="*50)
    print("Text Generation Examples")
    print("="*50)
    
    prompts = ["the company", "the stock market", "the president said"]
    
    print("\nVanilla RNN Generated Text:")
    for prompt in prompts:
        generated = generate_text(rnn_model, prompt, word2idx, idx2word, max_len=1)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}\n")
    
    print("\nLSTM Generated Text:")
    for prompt in prompts:
        generated = generate_text(lstm_model, prompt, word2idx, idx2word, max_len=+1)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}\n")


if __name__ == '__main__':
    main()
