import os
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import *
from utils import setup_gpu, set_random_seeds, augment_with_gaussian_noise

setup_gpu()
set_random_seeds(CV_CONFIG['random_state'])

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class GaborLike3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5, 5)):
        super(GaborLike3D, self).__init__()
        self.filters = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.theta = nn.Parameter(torch.empty(self.filters).uniform_(0, math.pi))
        self.sigma_x = nn.Parameter(torch.empty(self.filters).uniform_(1.0, 3.0))
        self.sigma_y = nn.Parameter(torch.empty(self.filters).uniform_(1.0, 3.0))
        self.sigma_z = nn.Parameter(torch.empty(self.filters).uniform_(1.0, 2.0))
        self.lambd = nn.Parameter(torch.empty(self.filters).uniform_(2.0, 8.0))
        self.gamma = nn.Parameter(torch.empty(self.filters).uniform_(0.5, 1.0))

    def generate_gabor_kernel(self):
        kz, ky, kx = self.kernel_size
        device = self.theta.device
        z = torch.arange(-(kz//2), kz//2 + 1, dtype=torch.float32, device=device)
        y = torch.arange(-(ky//2), ky//2 + 1, dtype=torch.float32, device=device)
        x = torch.arange(-(kx//2), kx//2 + 1, dtype=torch.float32, device=device)
        Z, Y, X = torch.meshgrid(z, y, x)
        Z, Y, X = Z.unsqueeze(-1), Y.unsqueeze(-1), X.unsqueeze(-1)
        theta = self.theta.view(1, 1, 1, -1)
        sigma_x = self.sigma_x.view(1, 1, 1, -1)
        sigma_y = self.sigma_y.view(1, 1, 1, -1)
        sigma_z = self.sigma_z.view(1, 1, 1, -1)
        lambd = self.lambd.view(1, 1, 1, -1)
        gamma = self.gamma.view(1, 1, 1, -1)
        x_theta = X * torch.cos(theta) + Y * torch.sin(theta)
        y_theta = -X * torch.sin(theta) + Y * torch.cos(theta)
        gaussian = torch.exp(-0.5 * (x_theta**2 / sigma_x**2 + gamma**2 * y_theta**2 / sigma_y**2 + Z**2 / sigma_z**2))
        cosine = torch.cos(2 * math.pi * x_theta / lambd)
        kernels = gaussian * cosine
        kernel_sum = torch.sum(torch.abs(kernels), dim=(0, 1, 2), keepdim=True)
        kernels = kernels / (kernel_sum + 1e-6)
        kernels = kernels.permute(3, 0, 1, 2)
        kernels = kernels.unsqueeze(1).repeat(1, self.in_channels, 1, 1, 1)
        return kernels

    def forward(self, inputs):
        kernels = self.generate_gabor_kernel()
        return F.conv3d(inputs, kernels, stride=1, padding=2)

class TokenFilteringLayer(nn.Module):
    def __init__(self, in_features, token_ratio=0.7, timesteps=12):
        super(TokenFilteringLayer, self).__init__()
        self.num_top_tokens = math.ceil(timesteps * token_ratio)
        self.importance_scorer = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features, key, value):
        scores = self.importance_scorer(features).squeeze(-1)
        _, top_indices = torch.topk(scores, k=self.num_top_tokens, dim=-1)
        expanded_indices = top_indices.unsqueeze(-1).expand(-1, -1, key.size(-1))
        selected_key = torch.gather(key, 1, expanded_indices)
        selected_value = torch.gather(value, 1, expanded_indices)
        return selected_key, selected_value

class NeuromorphicModule(nn.Module):
    def __init__(self, feat_from_cnn, cnn_config):
        super(NeuromorphicModule, self).__init__()
        self.gabor = GaborLike3D(in_channels=1, out_channels=32, kernel_size=(5, 5, 5))
        self.bn1 = nn.BatchNorm3d(32) if REGULARIZATION_CONFIG['use_batch_norm'] else nn.Identity()
        self.dropout1 = nn.Dropout(cnn_config['dropout_rate_1']) if REGULARIZATION_CONFIG['use_dropout'] else nn.Identity()
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64) if REGULARIZATION_CONFIG['use_batch_norm'] else nn.Identity()
        self.dropout2 = nn.Dropout(cnn_config['dropout_rate_2']) if REGULARIZATION_CONFIG['use_dropout'] else nn.Identity()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc_out = nn.Linear(64 * 2 * 4 * 4, feat_from_cnn)

    def forward(self, x):
        x = F.relu(self.bn1(self.gabor(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(x)
        x = self.flatten(x)
        return F.relu(self.fc_out(x))

class ChannelDecoupledModule(nn.Module):
    def __init__(self, feat_from_cnn):
        super(ChannelDecoupledModule, self).__init__()
        self.spatial_pathways = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.ReLU(),
                nn.BatchNorm3d(16),  
                nn.Conv3d(16, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1))
            ) for _ in range(5)
        ])
        self.fusion_conv = nn.Conv3d(16, 64, kernel_size=(5, 1, 1), padding=0)
        self.bn_fusion = nn.BatchNorm3d(64) if REGULARIZATION_CONFIG['use_batch_norm'] else nn.Identity()    
        self.conv2d = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2d = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_out = nn.Linear(128 * 4 * 4, feat_from_cnn)

    def forward(self, x):
        bands = torch.split(x, 1, dim=2)
        processed_bands = [pathway(band) for pathway, band in zip(self.spatial_pathways, bands)]
        x = torch.cat(processed_bands, dim=2)
        x = F.relu(x)
        x = self.fusion_conv(x)
        x = self.bn_fusion(x)  
        x = F.relu(x)
        x = x.squeeze(2)
        x = self.conv2d(x)
        x = F.relu(x)          
        x = self.pool2d(x)
        x = self.flatten(x)
        return F.relu(self.fc_out(x))

class AttentionBlock(nn.Module):
    def __init__(self, att_dim, num_heads):
        super(AttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=att_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(att_dim)

    def forward(self, query, key, value):
        query_t = query.transpose(0, 1)
        key_t = key.transpose(0, 1)
        value_t = value.transpose(0, 1)
        attn_output, _ = self.multihead_attn(query_t, key_t, value_t)
        attn_output = attn_output.transpose(0, 1)
        return self.layer_norm(query + attn_output)

class EmotionModel(nn.Module):
    def __init__(self, config):
        super(EmotionModel, self).__init__()
        self.timesteps = config['timesteps']
        self.att_dim = config['att_dim']
        self.global_module = ChannelDecoupledModule(config['feat_from_cnn'])
        self.hemispheric_module = NeuromorphicModule(config['feat_from_cnn'], CNN_CONFIG)
        self.global_q_proj = nn.Linear(config['feat_from_cnn'], self.att_dim)
        self.global_k_proj = nn.Linear(config['feat_from_cnn'], self.att_dim)
        self.global_v_proj = nn.Linear(config['feat_from_cnn'], self.att_dim)
        self.hemi_q_proj = nn.Linear(config['feat_from_cnn'], self.att_dim)
        self.hemi_k_proj = nn.Linear(config['feat_from_cnn'], self.att_dim)
        self.hemi_v_proj = nn.Linear(config['feat_from_cnn'], self.att_dim)
        self.global_lepe_conv = nn.Conv1d(self.att_dim, self.att_dim, kernel_size=5, padding=2, groups=self.att_dim)
        self.hemi_lepe_conv = nn.Conv1d(self.att_dim, self.att_dim, kernel_size=5, padding=2, groups=self.att_dim)
        self.hemi_token_filter = TokenFilteringLayer(config['feat_from_cnn'], timesteps=self.timesteps)
        self.global_token_filter = TokenFilteringLayer(config['feat_from_cnn'], timesteps=self.timesteps)
        self.attention_1 = AttentionBlock(self.att_dim, config['num_heads'])
        self.attention_2 = AttentionBlock(self.att_dim, config['num_heads'])
        self.gate_dense = nn.Linear(self.att_dim, self.att_dim)
        self.fc1 = nn.Linear(self.att_dim * 2, MLP_CONFIG['fc1_units'])
        self.dropout_fc1 = nn.Dropout(MLP_CONFIG['dropout_rate']) if REGULARIZATION_CONFIG['use_dropout'] else nn.Identity()
        self.fc2 = nn.Linear(MLP_CONFIG['fc1_units'], MLP_CONFIG['fc2_units'])
        self.dropout_fc2 = nn.Dropout(MLP_CONFIG['dropout_rate']) if REGULARIZATION_CONFIG['use_dropout'] else nn.Identity()
        self.output_layer = nn.Linear(MLP_CONFIG['fc2_units'], config['num_classes'])

    def forward(self, global_input, hemi_input):
        B, T, C, D, H, W = global_input.size()
        global_in_flat = global_input.view(B * T, C, D, H, W)
        hemi_in_flat = hemi_input.view(B * T, C, D, H, W)
        global_feat = self.global_module(global_in_flat).view(B, T, -1)
        hemi_feat = self.hemispheric_module(hemi_in_flat).view(B, T, -1)
        global_q = self.global_q_proj(global_feat)
        global_k = self.global_k_proj(global_feat)
        global_v = self.global_v_proj(global_feat)
        hemi_q = self.hemi_q_proj(hemi_feat)
        hemi_k = self.hemi_k_proj(hemi_feat)
        hemi_v = self.hemi_v_proj(hemi_feat)
        global_v_enh = global_v + self.global_lepe_conv(global_v.transpose(1, 2)).transpose(1, 2)
        hemi_v_enh = hemi_v + self.hemi_lepe_conv(hemi_v.transpose(1, 2)).transpose(1, 2)
        sel_hemi_k, sel_hemi_v = self.hemi_token_filter(hemi_feat, hemi_k, hemi_v_enh)
        sel_global_k, sel_global_v = self.global_token_filter(global_feat, global_k, global_v_enh)
        o1 = self.attention_1(global_q, sel_hemi_k, sel_hemi_v)
        o2 = self.attention_2(hemi_q, sel_global_k, sel_global_v)
        o1_pooled = o1.mean(dim=1)
        gate = torch.sigmoid(self.gate_dense(o1_pooled)).unsqueeze(1)
        o2_gated = o2 * gate
        concat_features = torch.cat([o1, o2_gated], dim=-1)
        final_features = concat_features.mean(dim=1)
        x = F.relu(self.fc1(final_features))
        x = self.dropout_fc1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        logits = self.output_layer(x)
        return logits

    def get_l2_loss(self, l2_strength):
        l2_loss = 0.0
        modules_with_l2 = [
            self.hemispheric_module.conv2,
            self.hemispheric_module.fc_out,
            self.global_module.fusion_conv,
            self.global_module.conv2d,
            self.global_module.fc_out,
            self.global_q_proj, self.global_k_proj, self.global_v_proj,
            self.hemi_q_proj, self.hemi_k_proj, self.hemi_v_proj,
            self.gate_dense, self.fc1, self.fc2
        ]
        for pathway in self.global_module.spatial_pathways:
            modules_with_l2.extend([pathway[0], pathway[3]])
        for module in modules_with_l2:
            l2_loss += torch.sum(module.weight ** 2)
        return l2_strength * l2_loss

class DataLoaderPyTorch:
    def __init__(self, data_dir, timesteps=12):
        self.data_dir = data_dir
        self.timesteps = timesteps
        self.global_data_path = os.path.join(data_dir, DATA_CONFIG['global_data_file'])
        self.hemispheric_data_path = os.path.join(data_dir, DATA_CONFIG['hemispheric_data_file'])
        self.label_path = os.path.join(data_dir, DATA_CONFIG['label_file'])
        self.full_global_data, self.full_hemispheric_data, self.full_labels = None, None, None

    def load_all_data(self):
        if self.full_global_data is None:
            self.full_global_data = np.load(self.global_data_path)
            self.full_hemispheric_data = np.load(self.hemispheric_data_path)
            self.full_labels = np.load(self.label_path)       

    def get_subject_data(self, subject_id):
        start_index = (subject_id - 1) * 3
        end_index = start_index + 3
        subject_global_raw = self.full_global_data[start_index:end_index]
        subject_hemispheric_raw = self.full_hemispheric_data[start_index:end_index]
        subject_global = subject_global_raw.reshape(-1, self.timesteps, 8, 9, 5)
        subject_hemispheric = subject_hemispheric_raw.reshape(-1, self.timesteps, 8, 9, 5)
        num_sequences_per_subject = self.full_global_data.shape[1] * 3
        subject_labels_raw = self.full_labels.reshape(DATA_CONFIG['num_subjects'], num_sequences_per_subject)[subject_id-1]
        subject_global = subject_global.transpose(0, 1, 4, 2, 3)
        subject_hemispheric = subject_hemispheric.transpose(0, 1, 4, 2, 3)
        subject_global = np.expand_dims(subject_global, axis=2)
        subject_hemispheric = np.expand_dims(subject_hemispheric, axis=2)
        label_mapping = {-1: 0, 0: 1, 1: 2}
        final_labels = np.array([label_mapping[lbl] for lbl in subject_labels_raw])
        return subject_global, subject_hemispheric, final_labels

class EEGExperiment:
    def __init__(self, data_dir=None, results_dir=None):
        self.data_dir = data_dir or DATA_CONFIG['data_dir']
        self.results_dir = results_dir or DATA_CONFIG['results_dir']
        self.data_loader = DataLoaderPyTorch(self.data_dir, MODEL_CONFIG['timesteps'])
        os.makedirs(self.results_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def run_subject_dependent_experiment(self, n_folds=10, start_subject=1):
        self.data_loader.load_all_data()
        all_results = []
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        txt_file = os.path.join(self.results_dir, f"results_{timestamp}.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Experiment initialized at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Folds: {n_folds} | Total Subjects: {DATA_CONFIG['num_subjects']}\n\n")
        for subject_id in range(start_subject, DATA_CONFIG['num_subjects'] + 1):
            global_data, hemispheric_data, labels = self.data_loader.get_subject_data(subject_id)
            subject_results, subject_logs = self.cross_validation(global_data, hemispheric_data, labels, subject_id, n_folds, timestamp)
            all_results.append(subject_results)
            res_str = f"Subject {subject_id} Results - Acc: {subject_results['mean_accuracy']:.4f}, F1: {subject_results['mean_f1']:.4f}"
            with open(txt_file, 'a', encoding='utf-8') as f:
                f.write('\n'.join(subject_logs) + '\n')
                f.write(res_str + '\n\n')
        self.save_final_summary(all_results, txt_file)
        return txt_file

    def get_lr_scheduler(self, optimizer, epoch):
        initial_lr, total_epochs, warmup_epochs, min_lr = TRAINING_CONFIG['learning_rate'], TRAINING_CONFIG['epochs'], TRAINING_CONFIG['warmup_epochs'], TRAINING_CONFIG['min_lr']
        if epoch < warmup_epochs:
            new_lr = initial_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs) if total_epochs != warmup_epochs else 1.0
            new_lr = min_lr + (initial_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def cross_validation(self, global_data, hemispheric_data, labels, subject_id, n_folds, timestamp):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=CV_CONFIG['shuffle'], random_state=CV_CONFIG['random_state'])
        fold_results = []
        subject_logs = [f"--- Subject {subject_id} ---"]
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            X_g_train, X_g_val = global_data[train_idx], global_data[val_idx]
            X_h_train, X_h_val = hemispheric_data[train_idx], hemispheric_data[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            if REGULARIZATION_CONFIG['data_augmentation']:
                X_g_train, X_h_train, y_train = augment_with_gaussian_noise(X_g_train, X_h_train, y_train, **AUGMENTATION_CONFIG)   
            train_dataset = TensorDataset(torch.FloatTensor(X_g_train), torch.FloatTensor(X_h_train), torch.LongTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_g_val), torch.FloatTensor(X_h_val), torch.LongTensor(y_val))
            train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True, drop_last=False)
            val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)
            model = EmotionModel(MODEL_CONFIG).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
            criterion = LabelSmoothingCrossEntropy(smoothing=TRAINING_CONFIG['label_smoothing'])
            checkpoint_filepath = f'./best_model_s{subject_id}_f{fold+1}_{timestamp}.pt'
            best_val_acc = 0.0
            patience_counter = 0
            for epoch in range(TRAINING_CONFIG['epochs']):
                model.train()
                self.get_lr_scheduler(optimizer, epoch)
                for bg, bh, by in train_loader:
                    bg, bh, by = bg.to(self.device), bh.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(bg, bh)
                    loss = criterion(outputs, by)
                    loss += model.get_l2_loss(REGULARIZATION_CONFIG['l2_strength'])
                    loss.backward()
                    optimizer.step()
                model.eval()
                val_preds, val_trues = [], []
                with torch.no_grad():
                    for bg, bh, by in val_loader:
                        bg, bh = bg.to(self.device), bh.to(self.device)
                        outputs = model(bg, bh)
                        preds = torch.argmax(outputs, dim=1)
                        val_preds.extend(preds.cpu().numpy())
                        val_trues.extend(by.numpy())
                current_val_acc = accuracy_score(val_trues, val_preds)
                if current_val_acc > best_val_acc:
                    best_val_acc = current_val_acc
                    torch.save(model.state_dict(), checkpoint_filepath)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= TRAINING_CONFIG['patience_early_stop']:
                        break 
            if os.path.exists(checkpoint_filepath):
                model.load_state_dict(torch.load(checkpoint_filepath, map_location=self.device))
            model.eval()
            final_preds = []
            with torch.no_grad():
                for bg, bh, _ in val_loader:
                    outputs = model(bg.to(self.device), bh.to(self.device))
                    final_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())    
            y_pred = np.array(final_preds)
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            fold_results.append({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1})
            subject_logs.append(f"Fold {fold+1}: Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
            if os.path.exists(checkpoint_filepath): 
                os.remove(checkpoint_filepath)
            del model
            torch.cuda.empty_cache()
        subject_metrics = {
            'subject_id': subject_id,
            'mean_accuracy': np.mean([r['accuracy'] for r in fold_results]), 
            'std_accuracy': np.std([r['accuracy'] for r in fold_results]),
            'mean_precision': np.mean([r['precision'] for r in fold_results]), 
            'std_precision': np.std([r['precision'] for r in fold_results]),
            'mean_recall': np.mean([r['recall'] for r in fold_results]), 
            'std_recall': np.std([r['recall'] for r in fold_results]),
            'mean_f1': np.mean([r['f1_score'] for r in fold_results]), 
            'std_f1': np.std([r['f1_score'] for r in fold_results]),
        }
        return subject_metrics, subject_logs
    
    def save_final_summary(self, all_results, txt_file):
        if not all_results: return
        mean_accuracies = [r['mean_accuracy'] for r in all_results]
        mean_f1s = [r['mean_f1'] for r in all_results]
        with open(txt_file, 'a', encoding='utf-8') as f:
            f.write("\n=== Overall Summary ===\n")
            f.write(f"Mean Acc : {np.mean(mean_accuracies):.4f} ± {np.std(mean_accuracies):.4f}\n")
            f.write(f"Mean F1  : {np.mean(mean_f1s):.4f} ± {np.std(mean_f1s):.4f}\n\n")
            f.write("Subject breakdown:\n")
            for r in all_results:
                f.write(f"Sub {r['subject_id']:2d} | Acc {r['mean_accuracy']:.4f} | F1 {r['mean_f1']:.4f}\n")
