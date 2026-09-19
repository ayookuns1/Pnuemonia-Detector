


import torch
import torch.nn as nn
import os
import utils, data_setup, engine, model_builder
from torchvision import transforms

train_dir = '/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray/train'
test_dir = '/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray/test'


device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir, test_dir, transform =data_transform, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS)


torch.manual_seed(42)
torch.cuda.manual_seed(42)


model = model_builder.XRayModel(
    input_shape = 3,
    hidden_units = 10,
    out_shape = len(class_names)
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)

NUM_EPOCHS = 10

from timeit import default_timer as timer

start_time = timer()

results = engine.train_model(model = model,
                     epochs = NUM_EPOCHS,
                     train_dataloader = train_dataloader, 
                     test_dataloader = test_dataloader,
                     loss_fn = loss_fn, optimizer = optimizer,
                     device = device)

end_time = timer()
print(f'Total time: {end_time - start_time}')


utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
