import pathlib
from EcoSet_Percept.ecoset_loader import EcoSetLoader

TARGET_CLASSES = [
    'car', 
    'house', 
    'phone', 
    'bed', 
    'cat', 
    'flower', 
    'train',
    'elephant', 
    'knife', 
    'bridge'
    ]

ecoset_dir = "/storage/coda1/p-drahnev6/0/shared/herrick/ecoset"
info_dir = pathlib.Path(__file__).parent.absolute() / 'EcoSet_Percept' / 'data'
save_path = 'EcoSet_Percept/data/ecoset_subset.npz'
loader = EcoSetLoader(ecoset_dir, info_dir, target_classes=TARGET_CLASSES)
loader.save_subset(n_test_img=50, save_path=save_path)