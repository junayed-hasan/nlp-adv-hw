#!/usr/bin/env python3
"""
ModernBERT LoRA Fine-tuning for StrategyQA Classification

This script implements LoRA (Low-Rank Adaptation) fine-tuning of ModernBERT on the StrategyQA dataset.
Uses the same number of trainable parameters as head-only training (~1,538 parameters).

IMPORTANT: Due to dataset script deprecation, install the correct datasets version:
    pip install "datasets<4.0.0"
    pip install peft>=0.7.0

Key features:
- LoRA fine-tuning with rank configuration to match head-only parameter count
- StrategyQA dataset: Binary classification task (True/False answers to strategy questions)  
- Modern best practices: Latest transformers library, mixed precision, proper initialization
- Early stopping: Prevents overfitting with patience-based monitoring
- Comprehensive evaluation: Accuracy, precision, recall, F1-score, confusion matrix
- Training visualization: Loss curves, accuracy plots, learning rate schedule
- Publication-quality plots: High-resolution figures suitable for academic papers
- Reproducible results: Fixed seeds and deterministic training
- Google Colab optimized: T4 GPU compatible with memory-efficient settings

Author: NLP Advanced Course
Date: 2024
"""

import os
import json
import time
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR

import transformers
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup,
    TrainingArguments, Trainer
)
from datasets import load_dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# PEFT for LoRA
from peft import LoraConfig, get_peft_model, TaskType

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA training of ModernBERT classifier."""
    
    # Model settings
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    num_labels: int = 2
    
    # LoRA configuration - tuned to match head-only parameter count (~1,538)
    lora_rank: int = 1  # Will be calculated to match target parameters
    lora_alpha: int = 1  # Lower scaling factor for minimal parameters
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # Will be set based on model architecture
    
    # Training hyperparameters
    learning_rate: float = 3e-4  # Lower LR for LoRA
    batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 15
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Early stopping
    early_stopping_patience: int = 5
    metric_for_best_model: str = "eval_accuracy"
    greater_is_better: bool = True
    
    # Technical settings
    fp16: bool = True  # Mixed precision for T4 GPU
    dataloader_num_workers: int = 2
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    
    # Random seeds
    seed: int = 42
    

class StrategyQADataset(Dataset):
    """Custom Dataset class for StrategyQA."""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get question text - handle different possible field names
        if 'question' in item:
            question = item['question']
        elif 'input' in item:
            question = item['input']
        elif 'text' in item:
            question = item['text']
        else:
            # Print available keys for debugging
            print(f"Warning: Could not find question field. Available keys: {list(item.keys())}")
            question = str(item)  # Fallback
        
        # Get label - handle different possible field names and formats
        if 'answer' in item:
            answer = item['answer']
            # Handle boolean, string, or integer answers
            if isinstance(answer, bool):
                label = 1 if answer else 0
            elif isinstance(answer, str):
                label = 1 if answer.lower() in ['true', 'yes', '1'] else 0
            elif isinstance(answer, int):
                label = answer
            else:
                label = 1 if answer else 0
        elif 'label' in item:
            label = item['label']
        elif 'target' in item:
            label = item['target']
        else:
            print(f"Warning: Could not find answer field. Available keys: {list(item.keys())}")
            label = 0  # Default fallback
        
        # Tokenize
        encoding = self.tokenizer(
            question,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class EarlyStoppingCallback:
    """Early stopping callback for training."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0001, metric: str = "eval_accuracy"):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        
    def __call__(self, current_score: float, epoch: int) -> bool:
        """
        Returns True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
        elif current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_strategyqa_dataset() -> DatasetDict:
    """Load and prepare StrategyQA dataset."""
    print("Loading StrategyQA dataset...")
    
    try:
        # Try with trust_remote_code for older datasets versions
        dataset = load_dataset("wics/strategy-qa", trust_remote_code=True)
        print(f"Dataset loaded successfully from wics/strategy-qa")
    except Exception as e:
        print(f"Error with trust_remote_code: {e}")
        print("Trying without trust_remote_code...")
        try:
            # Fallback without trust_remote_code
            dataset = load_dataset("wics/strategy-qa")
            print(f"Dataset loaded successfully from wics/strategy-qa")
        except Exception as e2:
            print(f"Both attempts failed: {e2}")
            raise RuntimeError(f"Could not load StrategyQA dataset. Error: {e2}")
    
    print(f"Available splits: {list(dataset.keys())}")
    
    # Print dataset info
    for split_name, split_data in dataset.items():
        print(f"  {split_name.capitalize()}: {len(split_data)} samples")
    
    return dataset


def process_predictions(predictions):
    """Helper function to consistently process predictions."""
    # Handle nested predictions structure
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Convert to numpy array if needed
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    
    # Ensure predictions are 2D
    if len(predictions.shape) == 1:
        # If 1D, assume binary classification and create 2D
        predictions = np.column_stack([1 - predictions, predictions])
    
    return predictions


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    
    # Process predictions consistently
    predictions = process_predictions(predictions)
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class ModernBERTLoRAClassifier(nn.Module):
    """ModernBERT with LoRA and classification head."""
    
    def __init__(self, model_name: str, num_labels: int = 2, lora_rank: int = 1):
        super().__init__()
        
        # Load base model
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Add classification head
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        # Apply LoRA to backbone
        target_modules = ["Wo"]  # Start with minimal modules
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,  # Keep alpha = rank for minimal scaling
            lora_dropout=0.1,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        self.backbone = get_peft_model(self.backbone, lora_config)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Forward through LoRA backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return {
            'loss': loss,
            'logits': logits,
            'last_hidden_state': outputs.last_hidden_state
        }
    
    def get_trainable_parameters(self):
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self):
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())


def setup_lora_model(model_name: str, config: LoRATrainingConfig):
    """Setup ModernBERT model with LoRA configuration."""
    
    print(f"Loading base model: {model_name}")
    
    # Create base model first to inspect structure
    base_model = AutoModel.from_pretrained(model_name)
    
    # Inspect module structure to find exact module names
    print("Inspecting model structure for specific modules...")
    wo_modules = []
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith('.Wo'):
            wo_modules.append(name)
    
    print(f"Found Wo modules: {wo_modules[:5]}...")  # Show first 5
    
    # Calculate expected parameters for different numbers of modules
    # Each Wo module with rank=1: (input_dim + output_dim) * rank
    # For ModernBERT-base: typically 768 input, 768 output -> 1536 params per module
    
    target_params = 1538  # Match head-only training
    
    # Strategy: Apply LoRA to only 1 specific Wo module to get close to target
    # Since each Wo module adds ~1536 params with rank=1, we target exactly 1 module
    
    if len(wo_modules) > 0:
        # Use only the first Wo module (layer 0)
        selected_modules = [wo_modules[0].split('.')[-1]]  # Just "Wo"
        
        # But we need to be more specific - target only specific layers
        # Instead of all "Wo", target specific module paths
        specific_target = wo_modules[0]  # e.g., "layers.0.attn.Wo"
        
        print(f"Targeting specific module: {specific_target}")
        
        # Create the LoRA model with specific module targeting
        model = AutoModel.from_pretrained(model_name)
        
        # Add classification head 
        model.classifier = nn.Linear(model.config.hidden_size, config.num_labels)
        nn.init.xavier_uniform_(model.classifier.weight)
        nn.init.zeros_(model.classifier.bias)
        
        # Try different approaches to limit LoRA parameters
        attempts = [
            # Attempt 1: Target only layer 0 Wo module using layers_to_transform
            {
                "target_modules": ["Wo"],
                "layers_to_transform": [0],  # Only apply to first layer
                "description": "layer 0 Wo only"
            },
            # Attempt 2: If layers_to_transform doesn't work, try specific pattern
            {
                "target_modules": ["layers.0.attn.Wo"],
                "description": "specific module path"
            },
            # Attempt 3: Fallback to all Wo but with very specific pattern
            {
                "target_modules": ["Wo"],
                "description": "all Wo modules (fallback)"
            }
        ]
        
        model_created = False
        for i, attempt in enumerate(attempts):
            try:
                print(f"Attempt {i+1}: Applying LoRA to {attempt['description']}")
                
                lora_config_kwargs = {
                    "r": 1,
                    "lora_alpha": 1,
                    "lora_dropout": 0.1,
                    "target_modules": attempt["target_modules"],
                    "bias": "none",
                    "task_type": TaskType.FEATURE_EXTRACTION,
                }
                
                # Add layers_to_transform if specified
                if "layers_to_transform" in attempt:
                    lora_config_kwargs["layers_to_transform"] = attempt["layers_to_transform"]
                
                lora_config = LoraConfig(**lora_config_kwargs)
                
                model = get_peft_model(model, lora_config)
                
                # Check parameter count
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  Trainable parameters: {trainable_params:,}")
                
                if trainable_params <= 2000:  # Close to our target
                    print(f"✓ Success! Parameter count acceptable.")
                    model_created = True
                    break
                else:
                    print(f"✗ Too many parameters ({trainable_params}), trying next approach...")
                    # Reset model for next attempt
                    model = AutoModel.from_pretrained(model_name)
                    model.classifier = nn.Linear(model.config.hidden_size, config.num_labels)
                    nn.init.xavier_uniform_(model.classifier.weight)
                    nn.init.zeros_(model.classifier.bias)
                    
            except Exception as e:
                print(f"✗ Attempt {i+1} failed: {e}")
                # Reset model for next attempt
                model = AutoModel.from_pretrained(model_name)
                model.classifier = nn.Linear(model.config.hidden_size, config.num_labels)
                nn.init.xavier_uniform_(model.classifier.weight)
                nn.init.zeros_(model.classifier.bias)
                continue
        
        if not model_created:
            raise RuntimeError("Could not create LoRA model with acceptable parameter count")
        
        # Calculate final parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nFinal LoRA Model:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.4f}%")
        print(f"Target was: {target_params}")
        print(f"Difference: {abs(trainable_params - target_params)}")
        
        # Wrap in our custom class for consistent interface
        class LoRAWrapper(nn.Module):
            def __init__(self, lora_model):
                super().__init__()
                self.model = lora_model
                
            def forward(self, input_ids, attention_mask, labels=None):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = self.model.classifier(outputs.last_hidden_state[:, 0])
                
                loss = None
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, labels)
                    
                return {
                    'loss': loss,
                    'logits': logits,
                    'last_hidden_state': outputs.last_hidden_state
                }
                
            def get_trainable_parameters(self):
                return sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            def get_total_parameters(self):
                return sum(p.numel() for p in self.parameters())
        
        return LoRAWrapper(model)
    
    else:
        raise RuntimeError("No Wo modules found in the model")


def plot_training_results(trainer_state, config: LoRATrainingConfig, save_dir: str = "."):
    """Create comprehensive training visualization plots."""
    
    # Extract training history
    log_history = trainer_state.log_history
    
    # Separate training and evaluation logs
    train_logs = [log for log in log_history if 'loss' in log and 'eval_loss' not in log]
    eval_logs = [log for log in log_history if 'eval_loss' in log]
    
    # Extract data for plotting
    train_epochs = [log['epoch'] for log in train_logs]
    train_loss = [log['loss'] for log in train_logs]
    train_lr = [log.get('learning_rate', 0) for log in train_logs]
    
    eval_epochs = [log['epoch'] for log in eval_logs]
    eval_loss = [log['eval_loss'] for log in eval_logs]
    eval_accuracy = [log['eval_accuracy'] for log in eval_logs]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ModernBERT LoRA Fine-tuning Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Training and Validation Loss
    ax1.plot(train_epochs, train_loss, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(eval_epochs, eval_loss, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    ax2.plot(eval_epochs, eval_accuracy, 'g-', label='Validation Accuracy', linewidth=2, marker='^', markersize=4)
    if eval_accuracy:
        best_acc = max(eval_accuracy)
        best_epoch = eval_epochs[eval_accuracy.index(best_acc)]
        ax2.axvline(x=best_epoch, color='orange', linestyle='--', alpha=0.7, label=f'Best Model (Epoch {best_epoch:.1f})')
        ax2.axhline(y=best_acc, color='orange', linestyle='--', alpha=0.7)
        ax2.text(best_epoch, best_acc + 0.01, f'{best_acc:.3f}', ha='center', va='bottom', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Learning Rate Schedule
    ax3.plot(train_epochs, train_lr, 'purple', linewidth=2, marker='d', markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 4: Training Progress Summary
    ax4.axis('off')
    
    # Summary statistics
    final_train_loss = train_loss[-1] if train_loss else 0
    final_val_loss = eval_loss[-1] if eval_loss else 0
    final_val_acc = eval_accuracy[-1] if eval_accuracy else 0
    best_val_acc = max(eval_accuracy) if eval_accuracy else 0
    
    summary_text = f"""
    Training Summary
    ════════════════
    
    Model: {config.model_name.split('/')[-1]} + LoRA
    Training Strategy: LoRA fine-tuning
    Dataset: StrategyQA (Binary Classification)
    
    LoRA Configuration:
    • Rank: {config.lora_rank}
    • Alpha: {config.lora_alpha}
    • Dropout: {config.lora_dropout}
    • Target Modules: Attention + FFN
    
    Hyperparameters:
    • Learning Rate: {config.learning_rate}
    • Batch Size: {config.batch_size}
    • Max Epochs: {config.num_epochs}
    • Early Stopping: {config.early_stopping_patience} epochs
    
    Final Results:
    • Final Training Loss: {final_train_loss:.4f}
    • Final Validation Loss: {final_val_loss:.4f}
    • Final Validation Accuracy: {final_val_acc:.4f}
    • Best Validation Accuracy: {best_val_acc:.4f}
    
    Training completed successfully!
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.1))
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig(os.path.join(save_dir, 'modernbert_lora_training_progress.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(save_dir, 'modernbert_lora_training_progress.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"LoRA training plots saved to {save_dir}/modernbert_lora_training_progress.png/pdf")


def plot_confusion_matrix(y_true, y_pred, save_dir: str = "."):
    """Plot confusion matrix for test results."""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['False', 'True'], 
                yticklabels=['False', 'True'])
    plt.title('Confusion Matrix - LoRA Test Set Results', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Answer')
    plt.ylabel('True Answer')
    
    # Add accuracy text
    accuracy = accuracy_score(y_true, y_pred)
    plt.text(0.5, -0.1, f'Test Accuracy: {accuracy:.4f}', 
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lora_confusion_matrix.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(save_dir, 'lora_confusion_matrix.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"LoRA confusion matrix saved to {save_dir}/lora_confusion_matrix.png/pdf")


def main():
    """Main training function."""
    
    print("ModernBERT LoRA Fine-tuning for StrategyQA Classification")
    print("==========================================================")
    
    # Set environment variables to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_SILENT"] = "true"
    
    # Initialize configuration
    config = LoRATrainingConfig()
    
    # Set random seed
    set_seed(config.seed)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load dataset
    dataset = load_strategyqa_dataset()
    
    # Handle different split naming conventions
    available_splits = list(dataset.keys())
    print(f"Available dataset splits: {available_splits}")
    
    # Map splits to standard names
    train_split = None
    val_split = None
    test_split = None
    
    for split in available_splits:
        if split in ['train', 'training']:
            train_split = split
        elif split in ['validation', 'valid', 'dev', 'val']:
            val_split = split
        elif split in ['test', 'testing']:
            test_split = split
    
    # If we don't have a validation split, create one from training data
    if train_split and not val_split:
        print("No validation split found, splitting training data...")
        train_data = dataset[train_split]
        train_size = int(0.8 * len(train_data))
        
        # Split training data
        train_data = train_data.select(range(train_size))
        val_data = dataset[train_split].select(range(train_size, len(dataset[train_split])))
        
    else:
        train_data = dataset[train_split] if train_split else dataset[available_splits[0]]
        val_data = dataset[val_split] if val_split else dataset[available_splits[1]] if len(available_splits) > 1 else train_data.select(range(100))
    
    test_data = dataset[test_split] if test_split else val_data
    
    # Create dataset objects
    train_dataset = StrategyQADataset(train_data, tokenizer, config.max_length)
    val_dataset = StrategyQADataset(val_data, tokenizer, config.max_length)
    test_dataset = StrategyQADataset(test_data, tokenizer, config.max_length)
    
    print("Dataset splits created:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Inspect a sample to verify data structure
    print("\nInspecting sample data...")
    try:
        sample = train_dataset[0]
        print(f"Sample input_ids shape: {sample['input_ids'].shape}")
        print(f"Sample attention_mask shape: {sample['attention_mask'].shape}")
        print(f"Sample label: {sample['labels'].item()}")
        
        # Decode a sample to verify tokenization
        decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        print(f"Sample question (decoded): {decoded[:100]}...")
        
    except Exception as e:
        print(f"Error inspecting sample: {e}")
        print("This might indicate an issue with dataset structure.")
    
    # Setup LoRA model
    print("Setting up ModernBERT with LoRA...")
    model = setup_lora_model(config.model_name, config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./modernbert_lora_output',
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        logging_dir='./lora_logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb
        seed=config.seed,
    )
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        patience=config.early_stopping_patience,
        metric=config.metric_for_best_model
    )
    
    # Custom trainer with early stopping
    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.early_stopping = early_stopping
            
        def evaluate(self, *args, **kwargs):
            result = super().evaluate(*args, **kwargs)
            
            # Check early stopping
            current_score = result.get(config.metric_for_best_model, 0)
            current_epoch = self.state.epoch
            
            if self.early_stopping(current_score, current_epoch):
                print(f"Early stopping triggered at epoch {current_epoch}")
                print(f"Best score: {self.early_stopping.best_score:.4f} at epoch {self.early_stopping.best_epoch}")
                self.control.should_training_stop = True
                
            return result
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    print("Starting LoRA training...")
    start_time = time.time()
    
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    print(f"LoRA training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    # Get predictions for confusion matrix
    predictions = trainer.predict(test_dataset)
    
    # Process predictions using helper function
    pred_logits = process_predictions(predictions.predictions)
    y_pred = np.argmax(pred_logits, axis=1)
    y_true = predictions.label_ids
    
    # Print detailed results
    print("\n" + "="*50)
    print("FINAL LoRA RESULTS")
    print("="*50)
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test F1 Score: {test_results['eval_f1']:.4f}")
    print(f"Test Precision: {test_results['eval_precision']:.4f}")
    print(f"Test Recall: {test_results['eval_recall']:.4f}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['False', 'True']))
    
    # Create visualizations
    print("\nCreating LoRA visualizations...")
    plot_training_results(trainer.state, config)
    plot_confusion_matrix(y_true, y_pred)
    
    # Save results
    results = {
        'model_name': config.model_name,
        'dataset': 'StrategyQA',
        'training_strategy': 'LoRA',
        'lora_config': {
            'rank': config.lora_rank,
            'alpha': config.lora_alpha,
            'dropout': config.lora_dropout,
            'target_modules': model.peft_config['default'].target_modules if hasattr(model, 'peft_config') else "auto-detected"
        },
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'training_time_seconds': training_time,
        'config': {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'max_length': config.max_length,
            'early_stopping_patience': config.early_stopping_patience,
        },
        'test_results': {
            'accuracy': test_results['eval_accuracy'],
            'f1': test_results['eval_f1'],
            'precision': test_results['eval_precision'],
            'recall': test_results['eval_recall'],
        },
        'best_model_epoch': early_stopping.best_epoch,
        'best_validation_score': early_stopping.best_score,
    }
    
    with open('modernbert_lora_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nLoRA results saved to modernbert_lora_results.json")
    print(f"Best LoRA model saved in {training_args.output_dir}")
    
    # Table 1 formatted results for the paper
    print("\n" + "="*50)
    print("TABLE 1 RESULTS (Copy to LaTeX)")
    print("="*50)
    print("ModernBERT (LoRA) & StrategyQA & "
          f"{test_results['eval_accuracy']:.3f} & "
          f"{test_results['eval_precision']:.3f} & "
          f"{test_results['eval_recall']:.3f} & "
          f"{test_results['eval_f1']:.3f} \\\\")
    
    return results


if __name__ == "__main__":
    main()
