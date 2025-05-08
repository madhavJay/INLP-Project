import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from datetime import datetime
import wandb

# --- Configuration ---
FINANCIAL_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume'] # Removed 'Ticker'
TEXT_FEATURE_COLUMN = 'news_title'  # This will be renamed to 'News'
DATE_COLUMN = 'date'
# TARGET_COLUMN is 'Close', predicting the Close of the day after the window.
CSV_PATH = 'final_model_input_1000.csv'
NUM_TOP_TICKERS = 1 # Number of top tickers to process

# Model and Training Hyperparameters (can be adjusted)
BASE_WINDOW = 5
SCALING_FACTOR = 3
MAX_WINDOW = 10
BERT_MODEL_NAME = "bert-base-uncased"  # Changed from FinBERT to regular BERT
MAX_TEXT_LENGTH = 128
BATCH_SIZE = 128
NUM_EPOCHS = 30 # As in notebook, can be increased
LEARNING_RATE = 1e-4
TCN_CHANNELS = [64, 64]
LSTM_HIDDEN = 128
FC_HIDDEN = 64
DROPOUT_RATE = 0.3 # For the final dropout in FusionModel
TCN_DROPOUT = 0.2 # For TCN blocks
FIN_INPUT_DIM = len(FINANCIAL_FEATURES) # Updated based on new FINANCIAL_FEATURES


# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
n_gpu = torch.cuda.device_count()
if device.type == 'cuda' and n_gpu > 1:
    print(f"Number of GPUs available: {n_gpu}")


# --- Helper Functions (adapted from notebook) ---
def create_dynamic_windows(df, financial_features_list, base_window=5, scaling_factor=3, max_window=10):
    X_fin = []
    y = []
    dates = []
    # Compute rolling volatility (std) over the base window from the 'Close' column
    # Ensure 'Close' is present and numeric
    if 'Close' not in df.columns or not pd.api.types.is_numeric_dtype(df['Close']):
        raise ValueError("DataFrame must contain a numeric 'Close' column for volatility calculation.")

    roll_std = df['Close'].rolling(window=base_window, min_periods=base_window).std()
    roll_std = roll_std.fillna(roll_std.mean()) # Fill NaNs that occur at the beginning

    for i in range(base_window, len(df)): # Start from base_window to ensure enough data for first window & std calc
        current_volatility = roll_std.iloc[i]
        if pd.isna(current_volatility): # Handle potential NaN if fillna didn't cover it
            current_volatility = roll_std.mean() # Fallback if still NaN

        extra = int(current_volatility * scaling_factor)
        window_size = min(base_window + extra, max_window)

        if i - window_size < 0: # Ensure we don't go out of bounds
            continue

        # Window is from i-window_size to i-1. Target is at i.
        window_data = df.iloc[i-window_size:i][financial_features_list].values
        target_price = df.iloc[i]['Close'] # Target is the 'Close' price at the end of the current day, for next day prediction context
        
        X_fin.append(window_data)
        y.append(target_price)
        dates.append(df.index[i])
        
    return X_fin, np.array(y), dates

tokenizer_global = None # To be initialized once

def get_tokenizer(model_name):
    global tokenizer_global
    if tokenizer_global is None:
        tokenizer_global = AutoTokenizer.from_pretrained(model_name)
    return tokenizer_global

def tokenize_text(text, tokenizer_instance, max_length=128):
    if not text or pd.isna(text): # Handle empty or NaN text
        # Return zero tensors for input_ids and attention_mask
        return torch.zeros(max_length, dtype=torch.long), torch.zeros(max_length, dtype=torch.long)
    
    encoded = tokenizer_instance.encode_plus(
        str(text), # Ensure text is string
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)

def collate_fn(batch):
    fin_data_list, input_ids_list, attention_mask_list, labels_list = zip(*batch)
    
    # Pad financial data sequences (they are variable-length tensors)
    # Convert list of numpy arrays (if they are) to list of tensors first
    fin_data_tensors = [torch.as_tensor(x, dtype=torch.float32) for x in fin_data_list]
    fin_data_padded = torch.nn.utils.rnn.pad_sequence(fin_data_tensors, batch_first=True, padding_value=0.0)
    
    lengths = torch.tensor([x.shape[0] for x in fin_data_tensors]) # Lengths before padding
    
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    labels = torch.tensor(labels_list, dtype=torch.float32) # Ensure labels are tensor
    
    return fin_data_padded, lengths, input_ids, attention_mask, labels

# --- Dataset Class (adapted from notebook) ---
class StockNewsDataset(Dataset):
    def __init__(self, X_fin, text_inputs, y):
        # X_fin is a list of numpy arrays, convert to tensors here or in collate_fn
        self.X_fin = X_fin # Will be list of numpy arrays from create_dynamic_windows
        self.text_inputs = text_inputs  # list of (input_ids, attention_mask) tensors
        self.y = y # numpy array from create_dynamic_windows
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        fin_data = self.X_fin[idx] # This is a numpy array
        input_ids, attention_mask = self.text_inputs[idx] # These are tensors
        label = self.y[idx] # This is a scalar numpy float
        
        # Ensure fin_data is tensor for collate_fn if not already handled
        # Collate_fn will handle numpy to tensor conversion for fin_data
        return torch.tensor(fin_data, dtype=torch.float32), input_ids, attention_mask, torch.tensor(label, dtype=torch.float32)

# --- Model Classes (copied from notebook) ---
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        
        if out.size(-1) != res.size(-1): # Align sequence length if needed due to padding/kernel
            min_len = min(out.size(-1), res.size(-1))
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1)*dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class FinancialBranchTCN(nn.Module):
    def __init__(self, input_dim, tcn_channels, kernel_size=2, dropout=0.2):
        super(FinancialBranchTCN, self).__init__()
        self.tcn = TCN(input_dim, tcn_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(tcn_channels[-1], 64) # Output 64 features

    def forward(self, x, lengths): # lengths might not be directly used by TCN if padding handled
        # x: [batch, seq_len, input_dim] -> transpose to [batch, input_dim, seq_len]
        x = x.transpose(1, 2)
        y = self.tcn(x)
        # Global average pooling over the time dimension
        y = torch.mean(y, dim=2)
        out = torch.relu(self.fc(y))
        return out

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output: [batch, seq_len, hidden_dim]
        attn_weights = torch.softmax(self.attn(lstm_output), dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context

class TextBranch(nn.Module):
    def __init__(self, bert_model_name, lstm_hidden=128, lstm_layers=1, bidirectional=True):
        super(TextBranch, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters(): # Freeze BERT parameters
            param.requires_grad = False 
        
        bert_output_dim = self.bert.config.hidden_size
        self.lstm = nn.LSTM(bert_output_dim, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=bidirectional)
        
        attention_input_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        self.attention = Attention(attention_input_dim)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad(): # Ensure BERT is in no_grad mode if frozen
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use last_hidden_state from BERT
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        text_rep = self.attention(lstm_out)
        return text_rep

class FusionModel(nn.Module):
    def __init__(self, bert_model_name, fin_input_dim, tcn_channels, lstm_hidden=128, fc_hidden=64, dropout_val=0.3):
        super(FusionModel, self).__init__()
        self.text_branch = TextBranch(bert_model_name, lstm_hidden=lstm_hidden)
        # FinancialBranchTCN outputs 64 features
        self.fin_branch = FinancialBranchTCN(fin_input_dim, tcn_channels=tcn_channels, dropout=TCN_DROPOUT)
        
        # Text branch output: lstm_hidden * 2 (if bidirectional)
        # Financial branch output: 64 (hardcoded in FinancialBranchTCN's fc layer)
        fusion_dim = (lstm_hidden * 2) + 64 
        
        self.fc1 = nn.Linear(fusion_dim, fc_hidden)
        self.dropout = nn.Dropout(dropout_val)
        self.fc2 = nn.Linear(fc_hidden, 1) # Predict a single value (next day's close price)

    def forward(self, fin_data, fin_lengths, input_ids, attention_mask):
        text_feat = self.text_branch(input_ids, attention_mask)
        fin_feat = self.fin_branch(fin_data, fin_lengths) # fin_lengths might not be used explicitly by current TCN
        
        fused = torch.cat([text_feat, fin_feat], dim=1)
        x = torch.relu(self.fc1(fused))
        x = self.dropout(x)
        out = self.fc2(x)
        return out.squeeze(1) # Squeeze to match label shape [batch_size]

# --- Main Script Logic ---
def main():
    # 1. Load and Preprocess Data
    print(f"Loading data from {CSV_PATH}...")
    try:
        full_raw_data_df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: {CSV_PATH} not found.")
        return

    if 'Ticker' not in full_raw_data_df.columns:
        print("Error: 'Ticker' column not found in CSV. This column is required to identify top tickers.")
        return
    
    print(f"Identifying the top {NUM_TOP_TICKERS} tickers by data points...")
    ticker_counts = full_raw_data_df['Ticker'].value_counts()
    if ticker_counts.empty:
        print("Error: 'Ticker' column has no data.")
        return
    
    top_tickers = ticker_counts.nlargest(NUM_TOP_TICKERS).index.tolist()
    print(f"Top {NUM_TOP_TICKERS} tickers to process: {top_tickers}")

    bert_tokenizer = get_tokenizer(BERT_MODEL_NAME) # Initialize tokenizer once

    for current_ticker in top_tickers:
        # Initialize wandb run for each ticker
        run_name = f"{current_ticker}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project="stock_prediction_hyperparam_tuning", # Choose your project name
            name=run_name,
            config={
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "base_window": BASE_WINDOW,
                "scaling_factor": SCALING_FACTOR,
                "max_window": MAX_WINDOW,
                "bert_model_name": BERT_MODEL_NAME,
                "max_text_length": MAX_TEXT_LENGTH,
                "tcn_channels": TCN_CHANNELS,
                "lstm_hidden": LSTM_HIDDEN,
                "fc_hidden": FC_HIDDEN,
                "dropout_rate": DROPOUT_RATE,
                "tcn_dropout": TCN_DROPOUT,
                "fin_input_dim": FIN_INPUT_DIM,
                "ticker": current_ticker # Log current ticker as part of config
            }
        )

        print(f"\n--- Processing Ticker: {current_ticker} ---")

        raw_data_df = full_raw_data_df[full_raw_data_df['Ticker'] == current_ticker].copy()
        
        if raw_data_df.empty:
            print(f"No data found for ticker {current_ticker} after filtering. Skipping.")
            wandb.finish() # Ensure wandb run is closed
            continue
        
        print(f"Data points for {current_ticker}: {len(raw_data_df)}")

        # Convert DATE_COLUMN to datetime, set as index, sort.
        raw_data_df[DATE_COLUMN] = pd.to_datetime(raw_data_df[DATE_COLUMN], utc=True)
        raw_data_df.sort_values(DATE_COLUMN, inplace=True)
        raw_data_df.set_index(DATE_COLUMN, inplace=True)

        # Rename TEXT_FEATURE_COLUMN to 'News' for consistency with tokenize_text
        if TEXT_FEATURE_COLUMN in raw_data_df.columns:
            raw_data_df.rename(columns={TEXT_FEATURE_COLUMN: 'News'}, inplace=True)
        else:
            print(f"Error: Text feature column '{TEXT_FEATURE_COLUMN}' not found for ticker {current_ticker}. Skipping.")
            wandb.finish() # Ensure wandb run is closed
            continue

        # Ensure all required financial features are present
        missing_cols = [col for col in FINANCIAL_FEATURES if col not in raw_data_df.columns]
        if missing_cols:
            print(f"Error: Financial feature columns {missing_cols} not found for ticker {current_ticker}. Skipping.")
            wandb.finish() # Ensure wandb run is closed
            continue
        if 'News' not in raw_data_df.columns: # Check after renaming
            print(f"Error: 'News' column (from '{TEXT_FEATURE_COLUMN}') not found for ticker {current_ticker}. Skipping.")
            wandb.finish() # Ensure wandb run is closed
            continue

        # Drop rows with NaN in critical financial features or News before scaling/windowing
        # FINANCIAL_FEATURES here is the modified list (without 'Ticker')
        critical_cols_for_ticker = FINANCIAL_FEATURES + ['News']
        raw_data_df.dropna(subset=critical_cols_for_ticker, inplace=True)
        if raw_data_df.empty:
            print(f"Error: DataFrame for ticker {current_ticker} is empty after dropping NaNs. Skipping.")
            wandb.finish() # Ensure wandb run is closed
            continue
            
        data_for_scaling = raw_data_df.copy()

        # 2. Scale Financial Data (per ticker)
        print(f"Scaling financial data for {current_ticker}...")
        scaler = StandardScaler() # Initialize a new scaler for each ticker
        
        # 3. Time-Based Train-Test Split (per ticker)
        print(f"Splitting data for {current_ticker} into train and test sets...")
        # Ensure enough data for split and windowing
        if len(data_for_scaling) < BASE_WINDOW * 2 + MAX_WINDOW : # Heuristic for minimal data
            print(f"Insufficient data for ticker {current_ticker} to perform train/test split and windowing ({len(data_for_scaling)} rows). Skipping.")
            wandb.finish() # Ensure wandb run is closed
            continue

        split_index = int(len(data_for_scaling) * 0.8)
        train_df = data_for_scaling.iloc[:split_index].copy()
        test_df = data_for_scaling.iloc[split_index:].copy()

        if train_df.empty or test_df.empty:
            print(f"Error: Train or test dataframe is empty for ticker {current_ticker}. Skipping.")
            wandb.finish() # Ensure wandb run is closed
            continue
        
        # Fit scaler ONLY on training data of the current ticker
        train_df[FINANCIAL_FEATURES] = scaler.fit_transform(train_df[FINANCIAL_FEATURES])
        # Transform test data using the SAME scaler
        test_df[FINANCIAL_FEATURES] = scaler.transform(test_df[FINANCIAL_FEATURES])
        
        # 4. Create Dynamic Windows and Targets (per ticker)
        print(f"Creating dynamic windows for {current_ticker} train set...")
        X_fin_train, y_fin_train, dates_train = create_dynamic_windows(
            train_df, FINANCIAL_FEATURES, BASE_WINDOW, SCALING_FACTOR, MAX_WINDOW
        )
        print(f"Creating dynamic windows for {current_ticker} test set...")
        X_fin_test, y_fin_test, dates_test = create_dynamic_windows(
            test_df, FINANCIAL_FEATURES, BASE_WINDOW, SCALING_FACTOR, MAX_WINDOW
        )

        if not X_fin_train or not X_fin_test: # Check if lists are empty, not just the arrays inside
            print(f"Error: No windows created for ticker {current_ticker}. Check windowing parameters and data length. Skipping.")
            wandb.finish() # Ensure wandb run is closed
            continue
        if len(X_fin_train) == 0 or len(X_fin_test) == 0:
             print(f"Error: Zero windows created for ticker {current_ticker}. Skipping.")
             wandb.finish() # Ensure wandb run is closed
             continue


        # 5. Tokenize Text Data (per ticker)
        print(f"Tokenizing text data for {current_ticker}...")
        # News aggregation for this ticker's data
        # Use `raw_data_df` which is already filtered for the current ticker and has date index
        if not isinstance(raw_data_df.index, pd.DatetimeIndex): # Should be already, but double check
            raw_data_df.index = pd.to_datetime(raw_data_df.index, utc=True)
        
        daily_news_aggregated_ticker = raw_data_df.groupby(raw_data_df.index.date)['News'].apply(lambda x: " ".join(x.dropna().astype(str)))
        daily_news_map_ticker = daily_news_aggregated_ticker.to_dict()

        texts_to_tokenize_train = [daily_news_map_ticker.get(d.date(), "") for d in dates_train]
        texts_to_tokenize_test = [daily_news_map_ticker.get(d.date(), "") for d in dates_test]

        print(f"Batch tokenizing {len(texts_to_tokenize_train)} train texts for {current_ticker}...")
        tokenized_outputs_train = bert_tokenizer(
            texts_to_tokenize_train, 
            add_special_tokens=True, 
            max_length=MAX_TEXT_LENGTH, 
            truncation=True, 
            padding='max_length', 
            return_tensors='pt'
        )
        print(f"Batch tokenizing {len(texts_to_tokenize_test)} test texts for {current_ticker}...")
        tokenized_outputs_test = bert_tokenizer(
            texts_to_tokenize_test, 
            add_special_tokens=True, 
            max_length=MAX_TEXT_LENGTH, 
            truncation=True, 
            padding='max_length', 
            return_tensors='pt'
        )

        text_inputs_train = [
            (tokenized_outputs_train['input_ids'][i], tokenized_outputs_train['attention_mask'][i]) 
            for i in range(len(texts_to_tokenize_train))
        ]
        text_inputs_test = [
            (tokenized_outputs_test['input_ids'][i], tokenized_outputs_test['attention_mask'][i]) 
            for i in range(len(texts_to_tokenize_test))
        ]

        # 6. Create Datasets and DataLoaders (per ticker)
        print(f"Creating datasets and dataloaders for {current_ticker}...")
        train_dataset = StockNewsDataset(X_fin_train, text_inputs_train, y_fin_train)
        test_dataset = StockNewsDataset(X_fin_test, text_inputs_test, y_fin_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        # 7. Initialize Model, Optimizer, Criterion (per ticker)
        print(f"Initializing model for {current_ticker}...")
        model = FusionModel(
            bert_model_name=BERT_MODEL_NAME,
            fin_input_dim=FIN_INPUT_DIM, # Already updated
            tcn_channels=TCN_CHANNELS,
            lstm_hidden=LSTM_HIDDEN,
            fc_hidden=FC_HIDDEN,
            dropout_val=DROPOUT_RATE
        ).to(device)
        
        force_single_gpu_debug = True 
        if not force_single_gpu_debug and device.type == 'cuda' and n_gpu > 1:
            print(f"Let's use {n_gpu} GPUs for {current_ticker}!")
            model = nn.DataParallel(model)
        elif device.type == 'cuda':
            print(f"Running on single GPU for {current_ticker}. Device: {device}")
        else:
            print(f"Running on CPU for {current_ticker}. Device: {device}")
        
        actual_model_for_check = model
        if not force_single_gpu_debug and isinstance(model, nn.DataParallel) and n_gpu > 1:
            actual_model_for_check = model.module
        
        print(f"Tokenizer vocab size (common): {bert_tokenizer.vocab_size}")
        print(f"Model BERT embedding vocab size for {current_ticker}: {actual_model_for_check.text_branch.bert.config.vocab_size}")

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        # 8. Training Loop (per ticker)
        print(f"Starting training for {current_ticker}...")
        for epoch in range(NUM_EPOCHS):
            model.train()
            epoch_losses = []
            model_to_get_vocab_from = model
            if not force_single_gpu_debug and isinstance(model, nn.DataParallel) and n_gpu > 1:
                model_to_get_vocab_from = model.module
            current_vocab_size = model_to_get_vocab_from.text_branch.bert.config.vocab_size
            
            for i, (fin_data, lengths, input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc=f"Ticker {current_ticker} - Epoch {epoch+1}/{NUM_EPOCHS}")):
                fin_data = fin_data.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                if input_ids.max().item() >= current_vocab_size or input_ids.min().item() < 0:
                    print(f"!!! Problematic input_ids in batch {i}, epoch {epoch+1} for ticker {current_ticker} !!!")
                    # ... (rest of error printing logic) ...
                    wandb.log({"error": f"Invalid input_ids for {current_ticker}. Max ID: {input_ids.max().item()}, Vocab Size: {current_vocab_size}"})
                    wandb.finish() # Finish run due to critical error
                    raise ValueError(f"Invalid input_ids for {current_ticker}. Max ID: {input_ids.max().item()}, Vocab Size: {current_vocab_size}")

                optimizer.zero_grad()
                outputs = model(fin_data, lengths.to(device), input_ids, attention_mask) 
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            print(f"Ticker {current_ticker} - Epoch {epoch+1}/{NUM_EPOCHS} - Training Loss: {avg_loss:.6f}")
            wandb.log({"epoch": epoch + 1, "training_loss": avg_loss}) # Log training loss

            # --- Per-Epoch Evaluation (per ticker) ---
            print(f"Evaluating model for {current_ticker} on test set after epoch {epoch+1}...")
            model.eval()
            epoch_all_preds = []
            epoch_all_labels = []
            with torch.no_grad():
                for fin_data_eval, lengths_eval, input_ids_eval, attention_mask_eval, labels_eval in tqdm(test_loader, desc=f"Evaluating {current_ticker} Epoch {epoch+1}"):
                    fin_data_eval = fin_data_eval.to(device)
                    input_ids_eval = input_ids_eval.to(device)
                    attention_mask_eval = attention_mask_eval.to(device)
                    
                    outputs_eval = model(fin_data_eval, lengths_eval.to(device), input_ids_eval, attention_mask_eval)
                    epoch_all_preds.extend(outputs_eval.cpu().numpy())
                    epoch_all_labels.extend(labels_eval.cpu().numpy())

            epoch_all_preds_np = np.array(epoch_all_preds)
            epoch_all_labels_np = np.array(epoch_all_labels)

            # Prepare a dictionary for wandb logging for this epoch's metrics
            epoch_metrics_to_log = {"epoch": epoch + 1}

            try:
                # Use the ticker-specific scaler
                close_idx = FINANCIAL_FEATURES.index('Close') # FINANCIAL_FEATURES is now the shorter list
                mean_close = scaler.mean_[close_idx]
                scale_close = scaler.scale_[close_idx]

                actual_unscaled_epoch = (epoch_all_labels_np * scale_close) + mean_close
                preds_unscaled_epoch = (epoch_all_preds_np * scale_close) + mean_close
                
                test_mse_unscaled_epoch = mean_squared_error(actual_unscaled_epoch, preds_unscaled_epoch)
                test_rmse_unscaled_epoch = np.sqrt(test_mse_unscaled_epoch)
                epoch_metrics_to_log["test_mse_unscaled"] = test_mse_unscaled_epoch
                epoch_metrics_to_log["test_rmse_unscaled"] = test_rmse_unscaled_epoch
                
                non_zero_actuals_mask_epoch = actual_unscaled_epoch != 0
                if np.any(non_zero_actuals_mask_epoch):
                    test_mape_unscaled_epoch = np.mean(np.abs((actual_unscaled_epoch[non_zero_actuals_mask_epoch] - preds_unscaled_epoch[non_zero_actuals_mask_epoch]) / actual_unscaled_epoch[non_zero_actuals_mask_epoch])) * 100
                    print(f"Ticker {current_ticker} - Epoch {epoch+1} - Test MAPE (unscaled): {test_mape_unscaled_epoch:.2f}%")
                    epoch_metrics_to_log["test_mape_unscaled"] = test_mape_unscaled_epoch
                else:
                    print(f"Ticker {current_ticker} - Epoch {epoch+1} - Test MAPE (unscaled): Cannot compute, all actual values are zero.")
                    epoch_metrics_to_log["test_mape_unscaled"] = np.nan # Log NaN if not computable

                print(f"Ticker {current_ticker} - Epoch {epoch+1} - Test MSE (unscaled): {test_mse_unscaled_epoch:.6f}")
                print(f"Ticker {current_ticker} - Epoch {epoch+1} - Test RMSE (unscaled): {test_rmse_unscaled_epoch:.6f}")

            except Exception as e:
                print(f"Ticker {current_ticker} - Epoch {epoch+1} - Could not calculate unscaled metrics: {e}")
                epoch_metrics_to_log["test_mse_unscaled_error"] = str(e)
                print(f"Ticker {current_ticker} - Epoch {epoch+1} - Using scaled metrics instead for error calculation.")
            
            test_mse_scaled_epoch = mean_squared_error(epoch_all_labels_np, epoch_all_preds_np)
            test_rmse_scaled_epoch = np.sqrt(test_mse_scaled_epoch)
            epoch_metrics_to_log["test_mse_scaled"] = test_mse_scaled_epoch
            epoch_metrics_to_log["test_rmse_scaled"] = test_rmse_scaled_epoch
            
            non_zero_actuals_mask_scaled_epoch = epoch_all_labels_np != 0
            if np.any(non_zero_actuals_mask_scaled_epoch):
                test_mape_scaled_epoch = np.mean(np.abs((epoch_all_labels_np[non_zero_actuals_mask_scaled_epoch] - epoch_all_preds_np[non_zero_actuals_mask_scaled_epoch]) / epoch_all_labels_np[non_zero_actuals_mask_scaled_epoch])) * 100
                print(f"Ticker {current_ticker} - Epoch {epoch+1} - Test MAPE (scaled): {test_mape_scaled_epoch:.2f}%")
                epoch_metrics_to_log["test_mape_scaled"] = test_mape_scaled_epoch
            else:
                print(f"Ticker {current_ticker} - Epoch {epoch+1} - Test MAPE (scaled): Cannot compute, all actual scaled values are zero.")
                epoch_metrics_to_log["test_mape_scaled"] = np.nan # Log NaN if not computable

            print(f"Ticker {current_ticker} - Epoch {epoch+1} - Test MSE (scaled): {test_mse_scaled_epoch:.6f}")
            print(f"Ticker {current_ticker} - Epoch {epoch+1} - Test RMSE (scaled): {test_rmse_scaled_epoch:.6f}")

            spearman_corr_epoch, _ = spearmanr(epoch_all_labels_np, epoch_all_preds_np)
            print(f"Ticker {current_ticker} - Epoch {epoch+1} - Spearman Correlation (scaled): {spearman_corr_epoch:.4f}")
            epoch_metrics_to_log["spearman_correlation_scaled"] = spearman_corr_epoch
            
            wandb.log(epoch_metrics_to_log) # Log all epoch metrics
        # --- End of Per-Epoch Evaluation ---

        # 9. Final Evaluation (for the current ticker, using last epoch's results)
        # The metrics from the last epoch serve as the final metrics for this run.
        # They are already logged.
        print(f"--- Final evaluation results for Ticker: {current_ticker} (from last epoch) ---")
        all_preds = epoch_all_preds_np # Results from the last epoch for current ticker
        all_labels = epoch_all_labels_np

        try:
            close_idx = FINANCIAL_FEATURES.index('Close')
            mean_close = scaler.mean_[close_idx] # Use current ticker's scaler
            scale_close = scaler.scale_[close_idx]

            actual_unscaled = (all_labels * scale_close) + mean_close
            preds_unscaled = (all_preds * scale_close) + mean_close
            
            test_mse_unscaled = mean_squared_error(actual_unscaled, preds_unscaled)
            test_rmse_unscaled = np.sqrt(test_mse_unscaled)
            non_zero_actuals_mask = actual_unscaled != 0
            if np.any(non_zero_actuals_mask):
                test_mape_unscaled = np.mean(np.abs((actual_unscaled[non_zero_actuals_mask] - preds_unscaled[non_zero_actuals_mask]) / actual_unscaled[non_zero_actuals_mask])) * 100
                print(f"Ticker {current_ticker} - Test MAPE (unscaled): {test_mape_unscaled:.2f}%")
            else:
                print(f"Ticker {current_ticker} - Test MAPE (unscaled): Cannot compute, all actual values are zero.")

            print(f"Ticker {current_ticker} - Test MSE (unscaled): {test_mse_unscaled:.6f}")
            print(f"Ticker {current_ticker} - Test RMSE (unscaled): {test_rmse_unscaled:.6f}")

        except Exception as e:
            print(f"Ticker {current_ticker} - Could not calculate unscaled metrics: {e}")
            print(f"Ticker {current_ticker} - Using scaled metrics instead.")
            test_mse = mean_squared_error(all_labels, all_preds)
            test_rmse = np.sqrt(test_mse)
            non_zero_actuals_mask_scaled = all_labels != 0
            if np.any(non_zero_actuals_mask_scaled):
                test_mape = np.mean(np.abs((all_labels[non_zero_actuals_mask_scaled] - all_preds[non_zero_actuals_mask_scaled]) / all_labels[non_zero_actuals_mask_scaled])) * 100
                print(f"Ticker {current_ticker} - Test MAPE (scaled): {test_mape:.2f}%")
            else:
                print(f"Ticker {current_ticker} - Test MAPE (scaled): Cannot compute, all actual scaled values are zero.")

            print(f"Ticker {current_ticker} - Test MSE (scaled): {test_mse:.6f}")
            print(f"Ticker {current_ticker} - Test RMSE (scaled): {test_rmse:.6f}")

        spearman_corr, _ = spearmanr(all_labels, all_preds)
        print(f"Ticker {current_ticker} - Spearman Correlation (scaled): {spearman_corr:.4f}")

        plt.figure(figsize=(12, 6))
        plt.plot(all_labels, label='Actual Scaled Close Prices')
        plt.plot(all_preds, label='Predicted Scaled Close Prices', alpha=0.7)
        plt.title(f'Test Set: Actual vs. Predicted Scaled Close Prices (Ticker: {current_ticker})')
        plt.xlabel('Time Steps (Test Set)')
        plt.ylabel('Scaled Close Price')
        plt.legend()
        plot_filename = f'predictions_vs_actual_scaled_{current_ticker}.png'
        plt.savefig(plot_filename)
        wandb.log({f"plot_predictions_vs_actual_scaled": wandb.Image(plot_filename)}) # Log plot
        print(f"Plot for {current_ticker} saved to {plot_filename}")
        # plt.show() # Uncomment if running in an environment that supports interactive plots
        plt.close() # Close the figure to free memory before the next ticker's plot
        
        wandb.finish() # Finish the wandb run for the current ticker

    print("\n--- All Ticker Processing Complete ---")

if __name__ == '__main__':
    main()