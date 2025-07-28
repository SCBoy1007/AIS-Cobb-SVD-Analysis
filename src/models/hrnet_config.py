# model/hrnet_config.py

MODEL_CONFIGS = {
    'hrnet18': {
        'STAGE1': {
            'NUM_MODULES': 1,
            'NUM_BRANCHES': 1,
            'BLOCK': 'BOTTLENECK',
            'NUM_BLOCKS': [4],
            'NUM_CHANNELS': [64],
            'FUSE_METHOD': 'SUM',
        },
        'STAGE2': {
            'NUM_MODULES': 1,
            'NUM_BRANCHES': 2,
            'BLOCK': 'BASIC',
            'NUM_BLOCKS': [4, 4],
            'NUM_CHANNELS': [18, 36],
            'FUSE_METHOD': 'SUM',
        },
        'STAGE3': {
            'NUM_MODULES': 4,
            'NUM_BRANCHES': 3,
            'BLOCK': 'BASIC',
            'NUM_BLOCKS': [4, 4, 4],
            'NUM_CHANNELS': [18, 36, 72],
            'FUSE_METHOD': 'SUM',
        },
        'STAGE4': {
            'NUM_MODULES': 3,
            'NUM_BRANCHES': 4,
            'BLOCK': 'BASIC',
            'NUM_BLOCKS': [4, 4, 4, 4],
            'NUM_CHANNELS': [18, 36, 72, 144],
            'FUSE_METHOD': 'SUM',
        },
    },
    'hrnet32': {
        'STAGE1': {
            'NUM_MODULES': 1,
            'NUM_BRANCHES': 1,
            'BLOCK': 'BOTTLENECK',
            'NUM_BLOCKS': [4],
            'NUM_CHANNELS': [64],
            'FUSE_METHOD': 'SUM',
        },
        'STAGE2': {
            'NUM_MODULES': 1,
            'NUM_BRANCHES': 2,
            'BLOCK': 'BASIC',
            'NUM_BLOCKS': [4, 4],
            'NUM_CHANNELS': [32, 64],  # 原始HRNet-32通道数
            'FUSE_METHOD': 'SUM',
        },
        'STAGE3': {
            'NUM_MODULES': 4,
            'NUM_BRANCHES': 3,
            'BLOCK': 'BASIC',
            'NUM_BLOCKS': [4, 4, 4],
            'NUM_CHANNELS': [32, 64, 128],  # 原始HRNet-32通道数
            'FUSE_METHOD': 'SUM',
        },
        'STAGE4': {
            'NUM_MODULES': 3,
            'NUM_BRANCHES': 4,
            'BLOCK': 'BASIC',
            'NUM_BLOCKS': [4, 4, 4, 4],
            'NUM_CHANNELS': [32, 64, 128, 256],  # 原始HRNet-32通道数
            'FUSE_METHOD': 'SUM',
        },
    },
}