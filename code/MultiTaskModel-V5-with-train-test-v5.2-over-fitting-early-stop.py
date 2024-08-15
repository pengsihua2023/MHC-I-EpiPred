import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import copy

class PathogenicDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        peptide = item['Peptide']
        inputs = self.tokenizer(peptide, return_tensors="pt", padding='max_length', max_length=50, truncation=True)
        inputs = {key: val.squeeze() for key, val in inputs.items()}
        labels = {
            'immunogenicity': torch.tensor([1 if item['Immunogenicity'] == 'Positive' else 0], dtype=torch.float)
        }
        return inputs, labels

    def collate_fn(self, batch):
        inputs = {key: torch.stack([b[0][key] for b in batch]) for key in batch[0][0]}
        labels = {key: torch.stack([b[1][key] for b in batch]) for key in batch[0][1]}
        return inputs, labels

class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
        self.dropout = nn.Dropout(0.3)  # Adding dropout layer
        self.classifier = nn.Linear(self.base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        pooled_output = self.dropout(pooled_output)  # Applying dropout
        immunogenicity = torch.sigmoid(self.classifier(pooled_output))
        return immunogenicity

def compute_loss(outputs, labels):
    criterion = nn.BCELoss()
    return criterion(outputs, labels['immunogenicity'].view(-1, 1))

def train_and_evaluate(model, train_loader, test_loader, optimizer, num_epochs, device):
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    patience = 10
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        model.train()
        total_loss, train_correct, train_total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = {k: v.to(device) for k, v in labels.items()}
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predicted = outputs.round()
            train_correct += (predicted == labels['immunogenicity']).sum().item()
            train_total += labels['immunogenicity'].numel()

        # Evaluate on test data
        test_loss, test_accuracy = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss / len(train_loader):.4f}, Train Acc: {100 * train_correct / train_total:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
        
        # Check for early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"No improvement in {patience} consecutive epochs, stopping early.")
                early_stop = True
                break

    # Load the best model weights
    if early_stop:
        model.load_state_dict(best_model_wts)
        print("Loaded best model weights!")
    
    return model

def evaluate(model, data_loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = {k: v.to(device) for k, v in labels.items()}
            outputs = model(**inputs)
            loss = compute_loss(outputs, labels)
            total_loss += loss.item()
            predicted = outputs.round()
            correct += (predicted == labels['immunogenicity']).sum().item()
            total += labels['immunogenicity'].numel()
    average_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return average_loss, accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv('pathogenic_db-5-colume-data.csv')
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    train_dataset = PathogenicDataset(train_data, tokenizer)
    test_dataset = PathogenicDataset(test_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=test_dataset.collate_fn)
    model = MultiTaskModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=0.05)  # L2 Regularization added

    model = train_and_evaluate(model, train_loader, test_loader, optimizer, 100, device)

if __name__ == '__main__':
    main()


