import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from datetime import datetime
# from sklearn.model_selection import train_test_split # Not strictly needed for simple chronological split

# --- Configuration ---
FINANCIAL_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
TEXT_FEATURE_COLUMN = 'news_title'
DATE_COLUMN = 'date'
NEW_TARGET_COLUMN = 'Next_Day_Close'
CSV_PATH = 'final_model_input_1000.csv' # MAKE SURE THIS PATH IS CORRECT

# Model and Training Hyperparameters
BASE_WINDOW = 5
SCALING_FACTOR = 3
MAX_WINDOW = 10
FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"
MAX_TEXT_LENGTH = 128
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 5e-5
TCN_CHANNELS = [64, 64]
LSTM_HIDDEN = 128
FC_HIDDEN = 64
DROPOUT_RATE = 0.3
TCN_DROPOUT = 0.2
# FIN_INPUT_DIM will be set dynamically after ticker processing
FIN_INPUT_DIM = len(FINANCIAL_FEATURES) # Initial value

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
n_gpu = torch.cuda.device_count()
if device.type == 'cuda' and n_gpu > 1:
    print(f"Number of GPUs available: {n_gpu}")

# --- Helper Functions ---
def create_dynamic_windows(df, financial_features_list, base_window=5, scaling_factor=3, max_window=10):
    X_fin = []
    y = []
    dates = []

    if 'Close' not in df.columns or not pd.api.types.is_numeric_dtype(df['Close']):
        raise ValueError("DataFrame must contain a numeric 'Close' column for volatility calculation.")
    if NEW_TARGET_COLUMN not in df.columns:
        raise ValueError(f"DataFrame must contain the target column '{NEW_TARGET_COLUMN}'.")

    roll_std = df['Close'].rolling(window=base_window, min_periods=base_window).std()
    mean_std_fill_val = roll_std.mean() # Calculate mean once for filling
    if pd.isna(mean_std_fill_val): # Handle case where all roll_std might be NaN (e.g. very short df)
        mean_std_fill_val = 0 # Default to 0 if no std can be calculated
    roll_std = roll_std.fillna(mean_std_fill_val)


    for i in range(base_window, len(df)):
        current_volatility = roll_std.iloc[i]
        if pd.isna(current_volatility): # Should be rare after fillna
            current_volatility = mean_std_fill_val

        extra = int(current_volatility * scaling_factor)
        window_size = min(base_window + extra, max_window)

        if i - window_size < 0:
            continue

        window_data = df.iloc[i-window_size : i][financial_features_list].values
        target_price = df.iloc[i][NEW_TARGET_COLUMN]
        
        X_fin.append(window_data)
        y.append(target_price)
        dates.append(df.index[i])
        
    return X_fin, np.array(y), dates

tokenizer_global = None
def get_tokenizer(model_name):
    global tokenizer_global
    if tokenizer_global is None:
        tokenizer_global = AutoTokenizer.from_pretrained(model_name)
    return tokenizer_global

def collate_fn(batch):
    fin_data_list, input_ids_list, attention_mask_list, labels_list = zip(*batch)
    
    fin_data_tensors = [torch.as_tensor(x, dtype=torch.float32) for x in fin_data_list]
    fin_data_padded = torch.nn.utils.rnn.pad_sequence(fin_data_tensors, batch_first=True, padding_value=0.0)
    
    lengths = torch.tensor([x.shape[0] for x in fin_data_tensors])
    
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    labels = torch.tensor(labels_list, dtype=torch.float32)
    
    return fin_data_padded, lengths, input_ids, attention_mask, labels

# --- Dataset Class ---
class StockNewsDataset(Dataset):
    def __init__(self, X_fin, text_inputs, y):
        self.X_fin = X_fin
        self.text_inputs = text_inputs
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        fin_data = self.X_fin[idx]
        input_ids, attention_mask = self.text_inputs[idx]
        label = self.y[idx]
        return torch.tensor(fin_data, dtype=torch.float32), input_ids, attention_mask, torch.tensor(label, dtype=torch.float32)

# --- Model Classes ---
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
        if out.size(-1) != res.size(-1):
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
        self.fc = nn.Linear(tcn_channels[-1], 64)

    def forward(self, x, lengths): # x: [batch, seq_len_padded, input_dim]
        x = x.transpose(1, 2) # -> [batch, input_dim, seq_len_padded]
        y_tcn = self.tcn(x)   # -> [batch, tcn_channels[-1], seq_len_tcn_out]

        batch_size, num_features_tcn, seq_len_tcn_out = y_tcn.shape
        
        mask = torch.arange(seq_len_tcn_out, device=x.device)[None, None, :] < lengths[:, None, None].to(x.device)
        mask = mask.float()

        y_masked = y_tcn * mask
        sum_masked_output = torch.sum(y_masked, dim=2)
        
        actual_lengths_for_pooling = torch.clamp(lengths.to(x.device).float(), min=1.0).unsqueeze(1)
        
        y_pooled = sum_masked_output / actual_lengths_for_pooling
        
        if torch.isnan(y_pooled).any() or torch.isinf(y_pooled).any():
            print("NaN or Inf detected in y_pooled in FinancialBranchTCN!")
            # Consider logging more details or raising an error for debugging
            # For example: torch.save({'y_tcn': y_tcn.cpu(), 'mask': mask.cpu(), 'lengths': lengths.cpu(), 'sum_masked_output': sum_masked_output.cpu(), 'actual_lengths_for_pooling': actual_lengths_for_pooling.cpu()}, 'debug_tcn.pt')
            # raise ValueError("NaN/Inf in TCN pooling")


        out = torch.relu(self.fc(y_pooled))
        return out

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attn(lstm_output), dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context

class TextBranch(nn.Module):
    def __init__(self, bert_model_name, lstm_hidden=128, lstm_layers=1, bidirectional=True):
        super(TextBranch, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False 
        
        bert_output_dim = self.bert.config.hidden_size
        self.lstm = nn.LSTM(bert_output_dim, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=bidirectional)
        attention_input_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        self.attention = Attention(attention_input_dim)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        text_rep = self.attention(lstm_out)
        return text_rep

class FusionModel(nn.Module):
    def __init__(self, bert_model_name, fin_input_dim, tcn_channels, lstm_hidden=128, fc_hidden=64, dropout_val=0.3):
        super(FusionModel, self).__init__()
        self.text_branch = TextBranch(bert_model_name, lstm_hidden=lstm_hidden)
        self.fin_branch = FinancialBranchTCN(fin_input_dim, tcn_channels=tcn_channels, dropout=TCN_DROPOUT)
        fusion_dim = (lstm_hidden * 2) + 64
        self.fc1 = nn.Linear(fusion_dim, fc_hidden)
        self.dropout = nn.Dropout(dropout_val)
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, fin_data, fin_lengths, input_ids, attention_mask):
        text_feat = self.text_branch(input_ids, attention_mask)
        fin_feat = self.fin_branch(fin_data, fin_lengths)
        fused = torch.cat([text_feat, fin_feat], dim=1)
        x = torch.relu(self.fc1(fused))
        x = self.dropout(x)
        out = self.fc2(x)
        return out.squeeze(1)

# --- Main Script Logic ---
def main():
    global FIN_INPUT_DIM # To modify global variable

    print(f"Loading data from {CSV_PATH}...")
    try:
        raw_data_df_orig = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: {CSV_PATH} not found.")
        return
    
    raw_data_df = raw_data_df_orig.copy() # Work with a copy

    current_financial_features = FINANCIAL_FEATURES.copy()
    if 'Ticker' in raw_data_df.columns:
        unique_tickers = raw_data_df['Ticker'].unique()
        print(f"Found tickers: {unique_tickers}. One-hot encoding...")
        if len(unique_tickers) > 1:
            ticker_dummies = pd.get_dummies(raw_data_df['Ticker'], prefix='Ticker', dtype=float)
            raw_data_df = pd.concat([raw_data_df, ticker_dummies], axis=1)
            current_financial_features.extend(ticker_dummies.columns.tolist())
        raw_data_df.drop('Ticker', axis=1, inplace=True)
    else:
        print("Warning: 'Ticker' column not found. Proceeding without ticker features.")

    FIN_INPUT_DIM = len(current_financial_features)
    print(f"Actual financial input dimension set to: {FIN_INPUT_DIM}")

    raw_data_df[DATE_COLUMN] = pd.to_datetime(raw_data_df[DATE_COLUMN], errors='coerce', utc=True)
    raw_data_df.dropna(subset=[DATE_COLUMN], inplace=True) # Drop rows where date conversion failed
    raw_data_df.sort_values(DATE_COLUMN, inplace=True)
    raw_data_df.set_index(DATE_COLUMN, inplace=True)

    # --- Time Gap Diagnostic (on original sorted data before shuffle) ---
    print("Checking for time gaps in the original sorted data index...")
    if not raw_data_df.empty:
        time_diffs = raw_data_df.index.to_series().diff()
        print("Summary of time differences between consecutive rows (in days):")
        print(time_diffs.dt.days.value_counts().sort_index().to_string()) # .to_string() for better printing
    else:
        print("Raw data is empty after date processing, cannot check time gaps.")
    # --- End Time Gap Diagnostic ---


    if TEXT_FEATURE_COLUMN in raw_data_df.columns:
        raw_data_df.rename(columns={TEXT_FEATURE_COLUMN: 'News'}, inplace=True)
    else:
        print(f"Error: Text feature column '{TEXT_FEATURE_COLUMN}' not found.")
        return

    required_cols_check = current_financial_features + [NEW_TARGET_COLUMN, 'News']
    for col in required_cols_check:
        if col not in raw_data_df.columns:
            print(f"Error: Required column '{col}' not found in DataFrame.")
            return
    
    raw_data_df.dropna(subset=required_cols_check, inplace=True)
    if raw_data_df.empty:
        print("Error: DataFrame is empty after dropping NaNs in critical columns.")
        return

    print("Shuffling data (then re-sorting by date for splitting)...")
    raw_data_df = raw_data_df.sample(frac=1, random_state=42)
    raw_data_df.sort_index(inplace=True) # Re-sort by date to maintain local temporal order for windowing

    print("Splitting data into train and test sets (before scaling)...")
    split_index = int(len(raw_data_df) * 0.8)
    train_df_unscaled = raw_data_df.iloc[:split_index].copy()
    test_df_unscaled = raw_data_df.iloc[split_index:].copy()
    
    if train_df_unscaled.empty or test_df_unscaled.empty:
        print("Error: Unscaled train or test dataframe is empty.")
        return

    print(f"Variance of UNSCALED '{NEW_TARGET_COLUMN}' in train_df_unscaled: {train_df_unscaled[NEW_TARGET_COLUMN].var():.4f}")
    if not test_df_unscaled.empty:
         print(f"Variance of UNSCALED '{NEW_TARGET_COLUMN}' in test_df_unscaled: {test_df_unscaled[NEW_TARGET_COLUMN].var():.4f}")

    plt.figure(figsize=(15,5))
    plt.plot(raw_data_df.index, raw_data_df[NEW_TARGET_COLUMN], label=f'Original {NEW_TARGET_COLUMN} (Shuffled-Resorted)', alpha=0.5, linestyle='-', marker='.', markersize=2)
    if not train_df_unscaled.empty:
        split_point_date_in_resorted_data = train_df_unscaled.index[-1]
        plt.axvline(split_point_date_in_resorted_data, color='r', linestyle='--', label=f'Train/Test Split ({split_point_date_in_resorted_data.date()})')
    plt.title(f'Original {NEW_TARGET_COLUMN} Over Time with Train/Test Split')
    plt.legend()
    plt.savefig('original_target_plot.png')
    print("Plot of original target saved to original_target_plot.png")

    print("Scaling financial data...")
    scaler = StandardScaler()
    COLUMNS_TO_SCALE = list(set(current_financial_features + [NEW_TARGET_COLUMN]))
    
    # Ensure columns to scale actually exist in the unscaled dataframes
    for df_check, df_name in [(train_df_unscaled, "train_df_unscaled"), (test_df_unscaled, "test_df_unscaled")]:
        for col_s in COLUMNS_TO_SCALE:
            if col_s not in df_check.columns:
                print(f"Error: Column '{col_s}' for scaling not found in {df_name}.")
                return

    train_df_scaled = train_df_unscaled.copy()
    test_df_scaled = test_df_unscaled.copy()
    
    scaler.fit(train_df_unscaled[COLUMNS_TO_SCALE])
    train_df_scaled.loc[:, COLUMNS_TO_SCALE] = scaler.transform(train_df_unscaled[COLUMNS_TO_SCALE])
    test_df_scaled.loc[:, COLUMNS_TO_SCALE] = scaler.transform(test_df_unscaled[COLUMNS_TO_SCALE])
    
    print("Creating dynamic windows for train set...")
    X_fin_train, y_fin_train, dates_train = create_dynamic_windows(
        train_df_scaled, current_financial_features, BASE_WINDOW, SCALING_FACTOR, MAX_WINDOW
    )
    print("Creating dynamic windows for test set...")
    X_fin_test, y_fin_test, dates_test = create_dynamic_windows(
        test_df_scaled, current_financial_features, BASE_WINDOW, SCALING_FACTOR, MAX_WINDOW
    )

    if not X_fin_train or (not X_fin_test and len(test_df_scaled) > base_window): # Allow empty test if test_df too small for any window
        print("Warning: No windows created for train or test set. Check data length and windowing parameters.")
        # Depending on strictness, you might want to return here if X_fin_train is empty.
        if not X_fin_train: return

    if len(y_fin_train) == 0:
        print("Error: y_fin_train is empty after windowing. Cannot proceed.")
        return
    # It's possible y_fin_test is empty if test_df_scaled is too short for windowing.
    # Only print variance if not empty.
    print(f"Variance of SCALED training targets (y_fin_train): {np.var(y_fin_train):.4f}")
    if len(y_fin_test) > 0:
        print(f"Variance of SCALED test targets (y_fin_test): {np.var(y_fin_test):.4f}")
    else:
        print("No test targets (y_fin_test is empty), possibly due to short test set after splitting/windowing.")


    # --- News Aggregation (uses original, non-shuffled data for date mapping) ---
    print("Aggregating daily news efficiently (from original data)...")
    news_aggregation_df = raw_data_df_orig.copy() # Use the very original loaded df
    news_aggregation_df[DATE_COLUMN] = pd.to_datetime(news_aggregation_df[DATE_COLUMN], errors='coerce', utc=True)
    news_aggregation_df.dropna(subset=[DATE_COLUMN], inplace=True)
    news_aggregation_df.sort_values(DATE_COLUMN, inplace=True)
    # No set_index needed here if groupby works on column
    
    if TEXT_FEATURE_COLUMN in news_aggregation_df.columns:
        news_aggregation_df.rename(columns={TEXT_FEATURE_COLUMN: 'News'}, inplace=True)
    else:
        print(f"Error: Text feature column '{TEXT_FEATURE_COLUMN}' not found for news aggregation.")
        return
    news_aggregation_df['News'] = news_aggregation_df['News'].fillna('')

    daily_news_aggregated = news_aggregation_df.groupby(news_aggregation_df[DATE_COLUMN].dt.date)['News'].apply(lambda x: " ".join(x.astype(str)))
    daily_news_map = daily_news_aggregated.to_dict()

    print("Preparing texts for batch tokenization...")
    bert_tokenizer = get_tokenizer(FINBERT_MODEL_NAME)
    texts_to_tokenize_train = [daily_news_map.get(d.date(), "") for d in dates_train]
    texts_to_tokenize_test = [daily_news_map.get(d.date(), "") for d in dates_test] if dates_test else []


    def batch_tokenize(texts, tokenizer, max_len):
        if not texts: # Handle empty list of texts
            return {'input_ids': torch.empty(0, max_len, dtype=torch.long), 
                    'attention_mask': torch.empty(0, max_len, dtype=torch.long)}
        return tokenizer(texts, add_special_tokens=True, max_length=max_len, truncation=True, padding='max_length', return_tensors='pt')

    tokenized_outputs_train = batch_tokenize(texts_to_tokenize_train, bert_tokenizer, MAX_TEXT_LENGTH)
    tokenized_outputs_test = batch_tokenize(texts_to_tokenize_test, bert_tokenizer, MAX_TEXT_LENGTH)

    text_inputs_train = [(tokenized_outputs_train['input_ids'][i], tokenized_outputs_train['attention_mask'][i]) for i in range(len(texts_to_tokenize_train))]
    text_inputs_test = [(tokenized_outputs_test['input_ids'][i], tokenized_outputs_test['attention_mask'][i]) for i in range(len(texts_to_tokenize_test))]
    
    # --- News Coverage Diagnostic ---
    num_train_days_with_news = sum(1 for text in texts_to_tokenize_train if text != "")
    num_test_days_with_news = sum(1 for text in texts_to_tokenize_test if text != "")
    print(f"Training: {num_train_days_with_news} out of {len(texts_to_tokenize_train)} samples have news ({num_train_days_with_news / len(texts_to_tokenize_train) * 100:.2f}% coverage).")
    if len(texts_to_tokenize_test) > 0:
        print(f"Test: {num_test_days_with_news} out of {len(texts_to_tokenize_test)} samples have news ({num_test_days_with_news / len(texts_to_tokenize_test) * 100:.2f}% coverage).")
    else:
        print("Test set has no text samples (texts_to_tokenize_test is empty).")
    # --- End News Coverage Diagnostic ---

    print("Creating datasets and dataloaders...")
    train_dataset = StockNewsDataset(X_fin_train, text_inputs_train, y_fin_train)
    # Only create test_dataset if there are test samples
    if X_fin_test and text_inputs_test and len(y_fin_test) > 0:
        test_dataset = StockNewsDataset(X_fin_test, text_inputs_test, y_fin_test)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory=device.type=='cuda')
    else:
        print("Skipping test dataloader creation as test data is insufficient or empty.")
        test_loader = None # Explicitly set to None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=device.type=='cuda')

    print("Initializing model...")
    model = FusionModel(
        bert_model_name=FINBERT_MODEL_NAME, fin_input_dim=FIN_INPUT_DIM, tcn_channels=TCN_CHANNELS,
        lstm_hidden=LSTM_HIDDEN, fc_hidden=FC_HIDDEN, dropout_val=DROPOUT_RATE
    ).to(device)
    
    force_single_gpu_debug = True
    if not force_single_gpu_debug and device.type == 'cuda' and n_gpu > 1:
        print(f"Using {n_gpu} GPUs!")
        model = nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True) # Shorter patience

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_losses = []
        epoch_grad_norms = []
        
        prog_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for i, (fin_data, lengths, input_ids, attention_mask, labels) in enumerate(prog_bar):
            fin_data, lengths, input_ids, attention_mask, labels = \
                fin_data.to(device), lengths.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            if i == 0 and epoch == 0:
                 print(f"Batch 0: fin_data shape: {fin_data.shape}, lengths min/max: {lengths.min().item()}/{lengths.max().item()}, input_ids shape: {input_ids.shape}")

            optimizer.zero_grad()
            outputs = model(fin_data, lengths, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Add gradient clipping

            total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            epoch_grad_norms.append(total_norm)

            optimizer.step()
            epoch_losses.append(loss.item())
            prog_bar.set_postfix(loss=loss.item(), avg_loss=np.mean(epoch_losses[-100:]) if epoch_losses else loss.item()) # Smoothed loss
        
        avg_loss = np.mean(epoch_losses)
        avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_loss:.6f}, Avg Grad Norm: {avg_grad_norm:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Evaluation for scheduler
        if test_loader: # Only evaluate if test_loader exists
            model.eval()
            val_losses = []
            with torch.no_grad():
                for fin_data, lengths, input_ids, attention_mask, labels in test_loader: # Use test_loader for validation loss
                    fin_data, lengths, input_ids, attention_mask, labels = \
                        fin_data.to(device), lengths.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
                    outputs = model(fin_data, lengths, input_ids, attention_mask)
                    val_loss = criterion(outputs, labels)
                    val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation Loss: {avg_val_loss:.6f}")
            scheduler.step(avg_val_loss)
        else: # If no test_loader, step scheduler based on training loss (less ideal)
            scheduler.step(avg_loss)


    if not test_loader:
        print("No test data to evaluate. Skipping final evaluation.")
        return

    print("Evaluating model on test set...")
    model.eval()
    all_preds, all_labels, test_losses_final = [], [], []
    with torch.no_grad():
        for fin_data, lengths, input_ids, attention_mask, labels in tqdm(test_loader, desc="Final Evaluating"):
            fin_data, lengths, input_ids, attention_mask, labels = \
                fin_data.to(device), lengths.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(fin_data, lengths, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            test_losses_final.append(loss.item())
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss_final = np.mean(test_losses_final)
    print(f"Final Average Test Loss: {avg_test_loss_final:.6f}")

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)

    if len(all_labels) == 0:
        print("No predictions/labels for final metrics calculation.")
        return

    try:
        target_idx_in_scaler = scaler.feature_names_in_.tolist().index(NEW_TARGET_COLUMN)
        mean_target, scale_target = scaler.mean_[target_idx_in_scaler], scaler.scale_[target_idx_in_scaler]
        actual_unscaled, preds_unscaled = (all_labels * scale_target) + mean_target, (all_preds * scale_target) + mean_target
        
        test_mse_unscaled, test_rmse_unscaled = mean_squared_error(actual_unscaled, preds_unscaled), np.sqrt(mean_squared_error(actual_unscaled, preds_unscaled))
        
        non_zero_mask = actual_unscaled != 0
        if np.any(non_zero_mask):
            mape_unscaled = np.mean(np.abs((actual_unscaled[non_zero_mask] - preds_unscaled[non_zero_mask]) / actual_unscaled[non_zero_mask])) * 100
            print(f"Test MAPE (unscaled): {mape_unscaled:.2f}%")
        else: print("Test MAPE (unscaled): Cannot compute, all actuals zero or filtered.")
        print(f"Test MSE (unscaled): {test_mse_unscaled:.6f}\nTest RMSE (unscaled): {test_rmse_unscaled:.6f}")

    except Exception as e:
        print(f"Could not calculate unscaled metrics: {e}. Using scaled metrics.")
        # Scaled metrics calculation (if unscaling fails or as primary)
        test_mse_scaled, test_rmse_scaled = mean_squared_error(all_labels, all_preds), np.sqrt(mean_squared_error(all_labels, all_preds))
        non_zero_mask_scaled = all_labels != 0
        if np.any(non_zero_mask_scaled):
            mape_scaled = np.mean(np.abs((all_labels[non_zero_mask_scaled] - all_preds[non_zero_mask_scaled]) / all_labels[non_zero_mask_scaled])) * 100
            print(f"Test MAPE (scaled): {mape_scaled:.2f}%")
        else: print("Test MAPE (scaled): Cannot compute, all scaled actuals zero or filtered.")
        print(f"Test MSE (scaled): {test_mse_scaled:.6f}\nTest RMSE (scaled): {test_rmse_scaled:.6f}")


    spearman_corr, _ = spearmanr(all_labels, all_preds)
    print(f"Spearman Correlation (scaled targets): {spearman_corr:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(all_labels, label='Actual Scaled Prices (Test Set)', marker='.', linestyle='-', markersize=3)
    plt.plot(all_preds, label='Predicted Scaled Prices (Test Set)', alpha=0.7, marker='x', linestyle='--', markersize=3)
    plt.title('Test Set: Actual vs. Predicted Scaled Prices')
    plt.xlabel('Time Steps (Test Set Samples)')
    plt.ylabel('Scaled Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('predictions_vs_actual_scaled.png')
    print("Plot saved to predictions_vs_actual_scaled.png")

if __name__ == '__main__':
    main()