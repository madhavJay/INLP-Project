import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import gc
import os
from torch.amp import GradScaler, autocast as autocast_amp # Use torch.amp
import math
import argparse # For command-line arguments
import wandb # Weights & Biases

# --- Configuration ---
# Data Paths
STOCK_PRICES_PATH = 'stock_prices_4.csv'
NEWS_ARTICLES_PATH = '/home/akshat/.cache/kagglehub/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests/versions/2/analyst_ratings_processed.csv'

# Model & Data Hyperparameters
FINANCIAL_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
TARGET_COLUMN = 'Next_Day_Close'
WINDOW_SIZE = 10

MAX_TITLES_PER_DAY = 5
MAX_TITLE_LEN = 64
# MODIFIED: Changed TEXT_MODEL_NAME to a FinBERT model
TEXT_MODEL_NAME = "ProsusAI/finbert" # Example FinBERT model

TICKER_EMBEDDING_DIM = 32
FINANCIAL_EMBEDDING_DIM = 64
# DAILY_NEWS_EMBEDDING_DIM is now determined by the text model inside TMD_Transformer

TRANSFORMER_ENCODER_LAYERS = 2
TRANSFORMER_NHEAD = 4 # FinBERT base has a hidden size of 768. Ensure d_model is divisible by nhead.
                      # The code already adjusts d_model if necessary.
TRANSFORMER_DIM_FEEDFORWARD = 512
FC_HIDDEN_1 = 256
FC_HIDDEN_2 = 128
DROPOUT_RATE = 0.2

# Training Hyperparameters
# These can be overridden by argparse or wandb sweeps
DEFAULT_BATCH_SIZE = 4 # Per GPU (FinBERT might be more memory intensive, may need to reduce)
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_WEIGHT_DECAY = 1e-6
DEFAULT_NUM_EPOCHS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Stock Price Prediction Training")
parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size per GPU')
parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate')
parser.add_argument('--epochs', type=int, default=DEFAULT_NUM_EPOCHS, help='Number of epochs')
parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY, help='Weight decay')
parser.add_argument('--load_model_path', type=str, default=None, help='Path to a saved model state_dict to load and continue training or evaluate')
parser.add_argument('--wandb_project', type=str, default="stock_prediction_tmd_finbert", help='WandB project name (updated for FinBERT)')
parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name (optional, otherwise auto-generated)')
parser.add_argument('--eval_only', action='store_true', help='If set, only run evaluation on the validation set (requires --load_model_path)')


# --- Data Loading and Preprocessing (Your latest working version) ---
def load_and_preprocess_data():
    # ... (Your full, working load_and_preprocess_data function) ...
    # (Copied from your previous version - ensure it's the correct one)
    print("Loading stock prices...")
    stock_df = pd.read_csv(STOCK_PRICES_PATH)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True).dt.normalize()

    if stock_df[TARGET_COLUMN].isnull().any():
        nan_target_count = stock_df[TARGET_COLUMN].isnull().sum()
        print(f"Warning: {nan_target_count} NaNs found in raw '{TARGET_COLUMN}' column.")
        stock_df.dropna(subset=[TARGET_COLUMN], inplace=True)
        print(f"Dropped {nan_target_count} rows with NaN targets. New shape: {stock_df.shape}")

    if stock_df.empty:
        raise ValueError("Stock DataFrame is empty after handling NaN targets.")

    AVAILABLE_TICKERS = sorted(stock_df['Ticker'].unique())
    if not AVAILABLE_TICKERS:
        raise ValueError("No tickers available after initial stock_df processing.")

    ticker_to_id = {ticker: i for i, ticker in enumerate(AVAILABLE_TICKERS)}
    num_tickers = len(AVAILABLE_TICKERS)
    print(f"Found {num_tickers} unique tickers in stock data after initial processing.")

    print("Loading news articles...")
    news_df = pd.read_csv(NEWS_ARTICLES_PATH, usecols=['title', 'date', 'stock'])
    news_df.rename(columns={'stock': 'Ticker', 'date': 'Date'}, inplace=True)
    news_df.dropna(subset=['title', 'Date', 'Ticker'], inplace=True)
    news_df = news_df[news_df['Ticker'].isin(AVAILABLE_TICKERS)]
    news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce', utc=True).dt.normalize()
    news_df.dropna(subset=['Date'], inplace=True)
    print(f"News data shape after filtering for available tickers: {news_df.shape}")

    print("Grouping news titles by ticker and date...")
    if not news_df.empty:
        news_grouped = news_df.groupby(['Ticker', 'Date'])['title'].apply(lambda x: list(x)[:MAX_TITLES_PER_DAY]).reset_index()
        print("Merging stock and news data...")
        data_df = pd.merge(stock_df, news_grouped, on=['Ticker', 'Date'], how='left')
    else:
        print("News data is empty after filtering. Proceeding with only stock data and empty titles.")
        data_df = stock_df.copy()
        data_df['title'] = [[] for _ in range(len(data_df))]

    data_df['title'] = data_df['title'].apply(lambda x: x if isinstance(x, list) else [])
    data_df.sort_values(by=['Ticker', 'Date'], inplace=True)
    data_df.reset_index(drop=True, inplace=True)

    print("Scaling financial features and target...")
    financial_scalers = {}
    target_scalers = {}
    processed_data_df = data_df.copy()

    for ticker in tqdm(AVAILABLE_TICKERS, desc="Scaling data per ticker"):
        ticker_indices = processed_data_df[processed_data_df['Ticker'] == ticker].index
        if ticker_indices.empty:
            continue

        current_financial_data = processed_data_df.loc[ticker_indices, FINANCIAL_FEATURES].values.astype(np.float32)
        all_nan_cols_fin = np.all(np.isnan(current_financial_data), axis=0)
        if np.any(all_nan_cols_fin):
            current_financial_data[:, all_nan_cols_fin] = 0.0

        scaler_fin = StandardScaler()
        try:
            scaler_fin.fit(current_financial_data)
            scaler_fin.scale_ = np.maximum(scaler_fin.scale_, 1e-7)
            scaled_financial_data = scaler_fin.transform(current_financial_data)
            scaled_financial_data = np.nan_to_num(scaled_financial_data, nan=0.0, posinf=0.0, neginf=0.0)
            processed_data_df.loc[ticker_indices, FINANCIAL_FEATURES] = scaled_financial_data
        except ValueError as e:
            print(f"Error scaling financial features for ticker {ticker}: {e}. Filling features with 0.")
            processed_data_df.loc[ticker_indices, FINANCIAL_FEATURES] = 0.0
        financial_scalers[ticker] = scaler_fin

        current_target_data = processed_data_df.loc[ticker_indices, [TARGET_COLUMN]].values.astype(np.float32)
        if np.all(np.isnan(current_target_data)):
            target_scalers[ticker] = None
            continue
        scaler_target = StandardScaler()
        try:
            scaler_target.fit(current_target_data)
            scaler_target.scale_ = np.maximum(scaler_target.scale_, 1e-7)
            scaled_target_data = scaler_target.transform(current_target_data)
            scaled_target_data = np.nan_to_num(scaled_target_data, nan=np.nan, posinf=np.nan, neginf=np.nan)
            processed_data_df.loc[ticker_indices, TARGET_COLUMN] = scaled_target_data
        except ValueError as e:
            print(f"Error scaling target for ticker {ticker}: {e}.")
        target_scalers[ticker] = scaler_target

    if processed_data_df[TARGET_COLUMN].isnull().any():
        nan_target_final_count = processed_data_df[TARGET_COLUMN].isnull().sum()
        print(f"Warning: {nan_target_final_count} NaNs still exist in '{TARGET_COLUMN}' after scaling. These samples will be skipped by the Dataset.")

    processed_data_df['ticker_id'] = processed_data_df['Ticker'].map(ticker_to_id)
    print(f"Preprocessing complete. Final DataFrame shape: {processed_data_df.shape}")
    return processed_data_df, ticker_to_id, num_tickers, financial_scalers, target_scalers

# --- Dataset and Model Classes (Your latest working versions) ---
class StockNewsDataset(Dataset):
    # ... (Your full, working StockNewsDataset class, ensuring it skips NaN targets) ...
    def __init__(self, df, ticker_to_id, window_size, financial_features, target_column, tokenizer, max_titles_per_day, max_title_len):
        self.df = df
        self.ticker_to_id = ticker_to_id
        self.window_size = window_size
        self.financial_features = financial_features
        self.target_column = target_column
        self.tokenizer = tokenizer
        self.max_titles_per_day = max_titles_per_day
        self.max_title_len = max_title_len
        self.samples = []
        self._create_samples()

    def _create_samples(self):
        for ticker_id_val, group in tqdm(self.df.groupby('ticker_id'), desc="Creating samples"): # Renamed ticker_id to avoid conflict
            group_values = group.reset_index(drop=True)
            if len(group_values) < self.window_size + 1:
                continue
            for i in range(self.window_size, len(group_values)):
                target_val = group_values.loc[i, self.target_column]
                if pd.isna(target_val): # Critical NaN target skip
                    continue

                financial_window_data = group_values.loc[i-self.window_size : i-1, self.financial_features].values.astype(np.float32)
                news_window_titles = []
                for j in range(i-self.window_size, i):
                    daily_titles = group_values.loc[j, 'title']
                    if len(daily_titles) < self.max_titles_per_day:
                        daily_titles.extend([""] * (self.max_titles_per_day - len(daily_titles)))
                    news_window_titles.append(daily_titles[:self.max_titles_per_day])

                target = target_val.astype(np.float32)
                current_ticker_id = group_values.loc[i, 'ticker_id']
                self.samples.append({
                    "financial_data": financial_window_data,
                    "news_titles": news_window_titles,
                    "ticker_id": current_ticker_id,
                    "target": target
                })
        print(f"Created {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        financial_data = torch.tensor(sample["financial_data"], dtype=torch.float)
        ticker_id = torch.tensor(sample["ticker_id"], dtype=torch.long)
        target = torch.tensor(sample["target"], dtype=torch.float)
        all_input_ids = []
        all_attention_masks = []
        for daily_titles_list in sample["news_titles"]:
            daily_input_ids = []
            daily_attention_masks = []
            for title_text in daily_titles_list:
                encoded = self.tokenizer(
                    title_text, add_special_tokens=True, max_length=self.max_title_len,
                    padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
                )
                daily_input_ids.append(encoded['input_ids'].squeeze(0))
                daily_attention_masks.append(encoded['attention_mask'].squeeze(0))
            all_input_ids.append(torch.stack(daily_input_ids))
            all_attention_masks.append(torch.stack(daily_attention_masks))
        news_input_ids = torch.stack(all_input_ids)
        news_attention_mask = torch.stack(all_attention_masks)
        return financial_data, news_input_ids, news_attention_mask, ticker_id, target

class DailyNewsAggregator(nn.Module):
    # ... (Your DailyNewsAggregator class) ...
    def __init__(self, transformer_hidden_dim, nhead=4, dropout=0.1): # nhead and dropout are not used by mean pooling
        super().__init__()
        pass

    def forward(self, title_embeddings):
        daily_news_embedding = title_embeddings.mean(dim=1)
        return daily_news_embedding

class TMD_Transformer(nn.Module):
    # ... (Your TMD_Transformer class, ensure window_size is used correctly) ...
    def __init__(self, num_financial_features, num_tickers, ticker_embedding_dim,
                 text_model_name, financial_embedding_dim, window_size,
                 transformer_encoder_layers, transformer_nhead, transformer_dim_feedforward,
                 fc_hidden_1, fc_hidden_2, dropout_rate): # Removed daily_news_embedding_dim from args
        super().__init__()
        self.window_size = window_size

        self.ticker_embedding = nn.Embedding(num_tickers, ticker_embedding_dim)
        print(f"Initializing text model: {text_model_name}")
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.actual_daily_news_embedding_dim = self.text_model.config.hidden_size
        print(f"Actual daily news embedding dim (from text model config): {self.actual_daily_news_embedding_dim}")

        self.daily_news_aggregator = DailyNewsAggregator(self.actual_daily_news_embedding_dim, nhead=transformer_nhead, dropout=dropout_rate)
        self.financial_processor = nn.Sequential(
            nn.Linear(num_financial_features, financial_embedding_dim * 2), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(financial_embedding_dim * 2, financial_embedding_dim)
        )

        self.combined_daily_dim_before_proj = self.actual_daily_news_embedding_dim + financial_embedding_dim + ticker_embedding_dim

        self.d_model = self.combined_daily_dim_before_proj
        if self.combined_daily_dim_before_proj % transformer_nhead != 0:
            self.d_model = (self.combined_daily_dim_before_proj // transformer_nhead + 1) * transformer_nhead
            self.input_projection = nn.Linear(self.combined_daily_dim_before_proj, self.d_model)
            print(f"Projecting combined_daily_dim from {self.combined_daily_dim_before_proj} to {self.d_model} for nhead compatibility.")
        else:
            self.input_projection = nn.Identity()
            print(f"Using combined_daily_dim {self.combined_daily_dim_before_proj} as d_model directly.")


        self.positional_encoding = nn.Parameter(torch.randn(1, self.window_size, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward, dropout=dropout_rate, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_encoder_layers)
        self.prediction_head = nn.Sequential(
            nn.Linear(self.d_model, fc_hidden_1), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_1, fc_hidden_2), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_2, 1)
        )

    def forward(self, financial_data, news_input_ids, news_attention_mask, ticker_ids):
        B, W, M, L_title = news_input_ids.shape

        ticker_embed = self.ticker_embedding(ticker_ids)
        ticker_embed_expanded = ticker_embed.unsqueeze(1).repeat(1, W, 1)

        news_input_ids_flat = news_input_ids.view(B * W * M, L_title)
        news_attention_mask_flat = news_attention_mask.view(B * W * M, L_title)
        text_outputs = self.text_model(input_ids=news_input_ids_flat, attention_mask=news_attention_mask_flat)
        title_embeddings_flat = text_outputs.last_hidden_state[:, 0, :] # CLS token embedding
        title_embeddings_daily_grouped = title_embeddings_flat.view(B * W, M, self.actual_daily_news_embedding_dim)
        daily_news_embed_flat = self.daily_news_aggregator(title_embeddings_daily_grouped)
        daily_news_embed = daily_news_embed_flat.view(B, W, self.actual_daily_news_embedding_dim)

        processed_financial_embed = self.financial_processor(financial_data)
        combined_daily_embed = torch.cat([daily_news_embed, processed_financial_embed, ticker_embed_expanded], dim=-1)
        projected_combined_embed = self.input_projection(combined_daily_embed)

        x = projected_combined_embed + self.positional_encoding[:, :W, :]

        transformer_output = self.temporal_transformer(x)
        last_day_output = transformer_output[:, -1, :]
        prediction = self.prediction_head(last_day_output)
        return prediction.squeeze(-1)


# --- Training and Evaluation Functions (Cleaned, with wandb logging) ---
def train_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch_num): # Added epoch_num for logging
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    num_batches_for_grad_norm = 0

    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch_num}", leave=False, dynamic_ncols=True)

    for batch_idx, (financial_data, news_ids, news_mask, ticker_ids, targets) in enumerate(progress_bar):
        financial_data = financial_data.to(device, non_blocking=True)
        news_ids = news_ids.to(device, non_blocking=True)
        news_mask = news_mask.to(device, non_blocking=True)
        ticker_ids = ticker_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            outputs = model(financial_data, news_ids, news_mask, ticker_ids)
            loss = criterion(outputs, targets)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"\n!!!! NaN/Inf DETECTED IN LOSS at batch {batch_idx} (Train Epoch {epoch_num}) !!!!")
            wandb.log({"train_batch_loss_raw": float('nan'), "epoch": epoch_num, "batch_idx": batch_idx})
            # Optionally, save inputs that caused NaN for debugging
            # torch.save({
            #     'financial_data': financial_data.cpu(),
            #     'news_ids': news_ids.cpu(),
            #     'news_mask': news_mask.cpu(),
            #     'ticker_ids': ticker_ids.cpu(),
            #     'targets': targets.cpu()
            # }, f"nan_input_batch_{epoch_num}_{batch_idx}.pt")
            return float('nan')

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        current_batch_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                current_batch_grad_norm += param_norm.item() ** 2
        current_batch_grad_norm = current_batch_grad_norm ** 0.5

        if not (math.isnan(current_batch_grad_norm) or math.isinf(current_batch_grad_norm)):
            total_grad_norm += current_batch_grad_norm
            num_batches_for_grad_norm +=1

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        avg_loss_so_far = total_loss / (batch_idx + 1)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss_so_far:.4f}", grad_norm=f"{current_batch_grad_norm:.2e}")

        # Log batch loss to wandb (optional, can be very verbose)
        # wandb.log({"train_batch_loss": loss.item(), "epoch": epoch_num, "batch_idx": batch_idx, "batch_grad_norm": current_batch_grad_norm})

    avg_epoch_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('nan')
    avg_epoch_grad_norm = total_grad_norm / num_batches_for_grad_norm if num_batches_for_grad_norm > 0 else float('nan')
    wandb.log({"train_epoch_loss": avg_epoch_loss, "avg_train_grad_norm": avg_epoch_grad_norm, "epoch": epoch_num})
    return avg_epoch_loss

def evaluate_epoch(model, dataloader, criterion, device, epoch_num): # Added epoch_num
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Evaluating Epoch {epoch_num}", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for batch_idx, (financial_data, news_ids, news_mask, ticker_ids, targets) in enumerate(progress_bar):
            financial_data = financial_data.to(device, non_blocking=True)
            news_ids = news_ids.to(device, non_blocking=True)
            news_mask = news_mask.to(device, non_blocking=True)
            ticker_ids = ticker_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(financial_data, news_ids, news_mask, ticker_ids)
                loss = criterion(outputs, targets)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"\n!!!! NaN/Inf DETECTED IN LOSS at batch {batch_idx} (Evaluation Epoch {epoch_num}) !!!!")
                wandb.log({"val_batch_loss_raw": float('nan'), "epoch": epoch_num, "batch_idx": batch_idx})
                return float('nan')

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{total_loss / (batch_idx + 1):.4f}")

    avg_epoch_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('nan')
    wandb.log({"val_epoch_loss": avg_epoch_loss, "epoch": epoch_num})
    return avg_epoch_loss

# --- Main Execution Block (Cleaned, with wandb and model loading) ---
if __name__ == '__main__':
    args = parser.parse_args()

    # --- WANDB Initialization ---
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name, # Optional: if None, wandb auto-generates a name
        config={ # Log hyperparameters
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size_per_gpu": args.batch_size,
            "weight_decay": args.weight_decay,
            "window_size": WINDOW_SIZE,
            "max_titles_per_day": MAX_TITLES_PER_DAY,
            "max_title_len": MAX_TITLE_LEN,
            "text_model_name": TEXT_MODEL_NAME, # This will now be "ProsusAI/finbert"
            "transformer_encoder_layers": TRANSFORMER_ENCODER_LAYERS,
            "dropout_rate": DROPOUT_RATE,
            # Add any other relevant hyperparameters
        }
    )
    # If using wandb sweeps, wandb.config will be populated by the sweep agent
    # You can then use wandb.config.lr, wandb.config.batch_size etc.
    # For now, we use args directly.
    current_lr = args.lr
    current_batch_size_per_gpu = args.batch_size
    current_epochs = args.epochs
    current_weight_decay = args.weight_decay

    # Load data
    data_df, ticker_to_id, num_tickers, financial_scalers, target_scalers = load_and_preprocess_data()
    data_df_to_use = data_df
    print(f"Loading tokenizer for: {TEXT_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    # Data Splitting
    unique_ticker_ids = data_df_to_use['ticker_id'].unique()
    test_size_actual = 0.2 if len(unique_ticker_ids) >= 2 else 0
    if len(unique_ticker_ids) == 0: raise ValueError("No unique tickers found.")

    train_ticker_ids, val_ticker_ids = train_test_split(unique_ticker_ids, test_size=test_size_actual, random_state=42) if test_size_actual > 0 else (unique_ticker_ids, np.array([]))

    train_df = data_df_to_use[data_df_to_use['ticker_id'].isin(train_ticker_ids)]
    val_df = data_df_to_use[data_df_to_use['ticker_id'].isin(val_ticker_ids)] if len(val_ticker_ids) > 0 else pd.DataFrame()
    print(f"Train DF shape: {train_df.shape}, Val DF shape: {val_df.shape}")
    if train_df.empty: raise ValueError("Train DataFrame is empty.")

    train_dataset = StockNewsDataset(train_df, ticker_to_id, WINDOW_SIZE, FINANCIAL_FEATURES, TARGET_COLUMN, tokenizer, MAX_TITLES_PER_DAY, MAX_TITLE_LEN)
    if len(train_dataset) == 0: raise ValueError("Train Dataset is empty.")

    val_loader = None
    if not val_df.empty:
        val_dataset = StockNewsDataset(val_df, ticker_to_id, WINDOW_SIZE, FINANCIAL_FEATURES, TARGET_COLUMN, tokenizer, MAX_TITLES_PER_DAY, MAX_TITLE_LEN)
        if len(val_dataset) > 0:
            eval_batch_size = current_batch_size_per_gpu
            if DEVICE.type == 'cuda' and torch.cuda.device_count() > 1:
                eval_batch_size = current_batch_size_per_gpu * torch.cuda.device_count()
            val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)
        else: print("Validation dataset created but is empty.")
    else: print("Validation DataFrame is empty.")

    # DataLoader for training
    actual_train_batch_size = current_batch_size_per_gpu
    num_gpus = 0
    if DEVICE.type == 'cuda':
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1: actual_train_batch_size = current_batch_size_per_gpu * num_gpus
    dataloader_num_workers = min(os.cpu_count() // 2 if os.cpu_count() else 2, actual_train_batch_size, 8) if actual_train_batch_size > 0 else 2
    # Consider reducing num_workers if OOM with FinBERT
    # dataloader_num_workers = 2 # A safer default for larger models
    train_loader = DataLoader(train_dataset, batch_size=actual_train_batch_size, shuffle=True, num_workers=dataloader_num_workers, pin_memory=True)
    print(f"Effective training batch size: {actual_train_batch_size}, Num workers: {dataloader_num_workers}")

    # Initialize model
    model = TMD_Transformer(
        num_financial_features=len(FINANCIAL_FEATURES), num_tickers=num_tickers,
        ticker_embedding_dim=TICKER_EMBEDDING_DIM, text_model_name=TEXT_MODEL_NAME, # Will use FinBERT
        financial_embedding_dim=FINANCIAL_EMBEDDING_DIM, window_size=WINDOW_SIZE,
        transformer_encoder_layers=TRANSFORMER_ENCODER_LAYERS, transformer_nhead=TRANSFORMER_NHEAD,
        transformer_dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD, fc_hidden_1=FC_HIDDEN_1,
        fc_hidden_2=FC_HIDDEN_2, dropout_rate=DROPOUT_RATE
    ).to(DEVICE)

    if args.load_model_path:
        print(f"Loading model from: {args.load_model_path}")
        try:
            state_dict = torch.load(args.load_model_path, map_location=DEVICE)
            # Check if the loaded state_dict is for a different text model
            # This is a basic check; more sophisticated checks might be needed if layers differ significantly
            loaded_text_model_dim = state_dict.get('text_model.embeddings.word_embeddings.weight.size', [None, None])[1]
            current_text_model_dim = model.text_model.config.hidden_size if hasattr(model, 'text_model') else None

            if loaded_text_model_dim is not None and current_text_model_dim is not None and \
               loaded_text_model_dim != current_text_model_dim:
                print(f"Warning: Loaded model's text embedding dimension ({loaded_text_model_dim}) "
                      f"differs from current model's ({current_text_model_dim}). "
                      "Text model weights might not load correctly. Consider training from scratch or fine-tuning.")
                # Optionally, remove text_model weights from state_dict if you want to reinitialize it
                # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('text_model.')}


            if isinstance(model, nn.DataParallel) and not all(k.startswith('module.') for k in state_dict.keys()):
                new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=False) # Use strict=False if text_model changed
            elif not isinstance(model, nn.DataParallel) and all(k.startswith('module.') for k in state_dict.keys()):
                new_state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
            print("Model loaded (strict=False due to potential text model change).")
        except Exception as e:
            print(f"Error loading model: {e}. Training from scratch.")
            if args.eval_only:
                print("Cannot run eval_only without a successfully loaded model.")
                wandb.finish()
                exit()


    if DEVICE.type == 'cuda' and num_gpus > 1 and not isinstance(model, nn.DataParallel):
        print(f"Wrapping model with nn.DataParallel for {num_gpus} GPUs.")
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=current_weight_decay)
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))

    m_ref = model.module if isinstance(model, nn.DataParallel) else model
    print(f"Model structure initialized. Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Model d_model: {m_ref.d_model}, Positional encoding shape: {m_ref.positional_encoding.shape}")

    # --- Evaluation Only Mode ---
    if args.eval_only:
        if not args.load_model_path:
            print("Error: --eval_only requires --load_model_path.")
            wandb.finish()
            exit()
        if val_loader:
            print("--- Running Evaluation Only ---")
            eval_loss = evaluate_epoch(model, val_loader, criterion, DEVICE, epoch_num=0)
            print(f"Evaluation Complete. Val Loss: {eval_loss:.6f}")
            wandb.summary["final_eval_loss"] = eval_loss
        else:
            print("No validation data to evaluate.")
        wandb.finish()
        exit()

    # --- Training Loop ---
    best_val_loss = float('inf')
    for epoch in range(1, current_epochs + 1):
        print(f"\n--- Epoch {epoch}/{current_epochs} ---")

        gc.collect()
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler, epoch)

        current_val_loss = float('inf')
        if val_loader:
            current_val_loss = evaluate_epoch(model, val_loader, criterion, DEVICE, epoch)
            print(f"Epoch {epoch} Summary: Train Loss: {train_loss:.6f}, Val Loss: {current_val_loss:.6f}")
        else:
            print(f"Epoch {epoch} Summary: Train Loss: {train_loss:.6f} (No validation)")

        if math.isnan(train_loss) or (val_loader and isinstance(current_val_loss, float) and math.isnan(current_val_loss)):
            print("NaN loss detected, stopping training.")
            wandb.log({"training_status": "stopped_nan_loss", "epoch": epoch})
            break

        checkpoint_name = f"tmd_transformer_finbert_epoch_{epoch}.pth" # Updated checkpoint name
        state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state_to_save, checkpoint_name)
        wandb.save(checkpoint_name)
        print(f"Saved model checkpoint: {checkpoint_name}")

        if val_loader and not math.isnan(current_val_loss) and current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_path = "tmd_transformer_finbert_best_model.pth" # Updated best model name
            torch.save(state_to_save, best_model_path)
            wandb.save(best_model_path)
            print(f"Saved new best model with Val Loss: {best_val_loss:.6f}")
            wandb.summary["best_val_loss"] = best_val_loss

    print("\nTraining complete.")
    if val_loader and best_val_loss != float('inf'):
        print(f"Best Validation Loss: {best_val_loss:.6f}")

    wandb.finish()