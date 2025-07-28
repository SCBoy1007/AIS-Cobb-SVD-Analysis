# Default configuration for VTF-18V model

class Config:
    # Model configuration
    input_h = 1536
    input_w = 512
    down_ratio = 4
    num_classes = 18
    backbone = 'hrnet18'  # or 'hrnet32'
    
    # Training configuration
    batch_size = 4
    epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5
    
    # Data configuration
    data_dir = 'dataset/data'
    labels_dir = 'dataset/labels'
    splits_dir = 'dataset/splits'
    
    # Output configuration
    save_dir = 'checkpoints'
    log_dir = 'logs'
    results_dir = 'results'
    
    # Hardware configuration
    gpu_ids = [0]
    num_workers = 4
    
    # Loss configuration
    heatmap_weight = 1.0
    vector_weight = 0.05
    consistency_weight = 0.0