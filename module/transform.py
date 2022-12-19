import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 


train_transform = A.Compose([        
    A.Normalize(mean=(0.359, 0.370, 0.361), std=(0.087, 0.086, 0.088)),          
    ToTensorV2()
])

test_transform = A.Compose([
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),        
    A.Normalize(mean=(0.359, 0.370, 0.361), std=(0.087, 0.086, 0.088)),           
    ToTensorV2()
])
