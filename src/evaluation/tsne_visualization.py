import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from pathlib import Path
from tqdm import tqdm


@torch.no_grad()
def extract_embeddings(model, dataloader, device, output_dir="outputs"):
    """Extract embeddings từ model"""
    model.eval()
    
    all_image_embeds = []
    all_text_embeds = []
    all_fused_embeds = []
    all_labels = []
    all_image_paths = []
    
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]
        metadata = batch.get("metadata")
        
        if metadata is not None:
            metadata = metadata.to(device)
        
        # Extract embeddings từ model
        image_feat = model.vision_encoder(images)
        image_feat = model.vision_projection(image_feat)
        
        text_feat = model.text_encoder(input_ids, attention_mask)
        text_feat = model.text_projection(text_feat)
        
        # Fused embedding (image + text)
        if model.fusion_type == "concat":
            fused = model.concat_fusion(image_feat, text_feat)
        else:
            image_feat_unsq = image_feat.unsqueeze(1)
            text_feat_unsq = text_feat.unsqueeze(1)
            fused_attn, _ = model.cross_attention(image_feat_unsq, text_feat_unsq)
            fused = fused_attn.squeeze(1)
        
        # Integrate metadata if available
        if model.use_metadata and metadata is not None:
            metadata_feat = model.metadata_encoder(metadata)
            if metadata_feat.shape[-1] != fused.shape[-1]:
                if not hasattr(model, 'temp_metadata_proj'):
                    model.temp_metadata_proj = torch.nn.Linear(
                        metadata_feat.shape[-1], fused.shape[-1]
                    ).to(device)
                metadata_feat = model.temp_metadata_proj(metadata_feat)
            fused = fused + metadata_feat
        
        all_image_embeds.append(image_feat.cpu().numpy())
        all_text_embeds.append(text_feat.cpu().numpy())
        all_fused_embeds.append(fused.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_image_paths.extend(batch["image_path"])
    
    return {
        'image_embeds': np.vstack(all_image_embeds),
        'text_embeds': np.vstack(all_text_embeds),
        'fused_embeds': np.vstack(all_fused_embeds),
        'labels': np.array(all_labels),
        'image_paths': all_image_paths
    }


def plot_tsne_embeddings(embeddings_dict, save_path, class_names):
    """Vẽ t-SNE cho tất cả các embeddings"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    embedding_types = [
        ('image_embeds', 'Image Embeddings (EfficientNet-B0)'),
        ('text_embeds', 'Text Embeddings (PhoBERT)'),
        ('fused_embeds', 'Fused Embeddings (Image + Text + Metadata)')
    ]
    
    for idx, (emb_key, title) in enumerate(embedding_types):
        embeds = embeddings_dict[emb_key]
        labels = embeddings_dict['labels']
        
        # Giảm số lượng samples nếu quá nhiều
        if len(embeds) > 1000:
            indices = np.random.choice(len(embeds), 1000, replace=False)
            embeds = embeds[indices]
            labels = labels[indices]
        
        # t-SNE - FIX: change n_iter to max_iter
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        embeds_2d = tsne.fit_transform(embeds)
        
        ax = axes[idx]
        scatter = ax.scatter(embeds_2d[:, 0], embeds_2d[:, 1], 
                            c=labels, cmap='tab10', alpha=0.7, s=30)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_ticks(range(len(class_names)))
        cbar.set_ticklabels(class_names)
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ t-SNE visualization saved to {save_path}")


def plot_embedding_separation(embeddings_dict, save_path, class_names):
    """Vẽ biểu đồ khoảng cách giữa các class embeddings"""
    from sklearn.metrics import pairwise_distances
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    embedding_types = [
        ('image_embeds', 'Image Space'),
        ('text_embeds', 'Text Space'),
        ('fused_embeds', 'Fused Space')
    ]
    
    for idx, (emb_key, title) in enumerate(embedding_types):
        embeds = embeddings_dict[emb_key]
        labels = embeddings_dict['labels']
        
        # Tính centroids của mỗi class
        centroids = []
        valid_classes = []
        for c in range(len(class_names)):
            mask = labels == c
            if mask.any():
                centroids.append(embeds[mask].mean(axis=0))
                valid_classes.append(class_names[c])
        
        if len(centroids) < 2:
            print(f"Warning: Not enough classes for {title}")
            axes[idx].text(0.5, 0.5, f'Not enough classes\n({len(centroids)} classes)', 
                          ha='center', va='center')
            continue
            
        centroids = np.array(centroids)
        distances = pairwise_distances(centroids)
        
        ax = axes[idx]
        im = ax.imshow(distances, cmap='RdYlGn_r', interpolation='nearest', 
                       vmin=0, vmax=distances.max() if distances.max() > 0 else 1)
        ax.set_xticks(range(len(valid_classes)))
        ax.set_yticks(range(len(valid_classes)))
        ax.set_xticklabels(valid_classes, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(valid_classes, fontsize=9)
        ax.set_title(f'Class Separation - {title}', fontsize=12, fontweight='bold')
        
        for i in range(len(valid_classes)):
            for j in range(len(valid_classes)):
                text_val = distances[i, j]
                text_color = "black" if text_val > distances.max() / 2 else "white"
                ax.text(j, i, f'{text_val:.2f}', ha="center", va="center", 
                       color=text_color, fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Euclidean Distance')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Embedding separation plot saved to {save_path}")


def plot_tsne_comparison(embeddings_dict, save_path, class_names):
    """Vẽ so sánh t-SNE giữa các lớp (mỗi lớp một màu riêng biệt)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    embedding_types = [
        ('image_embeds', 'Image Space'),
        ('text_embeds', 'Text Space'),
        ('fused_embeds', 'Fused Space')
    ]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for idx, (emb_key, title) in enumerate(embedding_types):
        embeds = embeddings_dict[emb_key]
        labels = embeddings_dict['labels']
        
        if len(embeds) > 1000:
            indices = np.random.choice(len(embeds), 1000, replace=False)
            embeds = embeds[indices]
            labels = labels[indices]
        
        try:
            # t-SNE - FIX: change n_iter to max_iter
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
            embeds_2d = tsne.fit_transform(embeds)
            
            ax = axes[idx]
            for c in range(len(class_names)):
                mask = labels == c
                if mask.any():
                    ax.scatter(embeds_2d[mask, 0], embeds_2d[mask, 1], 
                              c=[colors[c]], label=class_names[c], 
                              alpha=0.6, s=25)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error plotting {title}: {e}")
            axes[idx].text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ t-SNE comparison saved to {save_path}")