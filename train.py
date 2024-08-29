import torch
from torchvision import transforms
from dataloaders import get_dataloaders
from model.deeplabv3plus import DeepLabv3
from lrscheduler import CustomLRScheduler
from download_dataset import download_dataset
from engine import train
from utils import load_config, read_palette

config_data = load_config()

URL = config_data['url']
PATH = config_data['path']
SOURCE = PATH+config_data['source_path']
TARGET = PATH+config_data['target_path']
BATCH_SIZE = config_data['batch_size']
IMAGE_SIZE = (config_data['image_x'], config_data['image_y'])
L_R = config_data['l_r']
EPOCHS = config_data['epochs']
POWER = config_data['power']
TRAIN_SIZE = config_data['train_size']
CLASSES = config_data['classes']
IGNORE_INDEX = config_data['ignore_index']
MOMENTUM = config_data['momentum']
MODEL_NAME = config_data['name']
PALETTE_PATH = config_data['palette_path']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(IMAGE_SIZE, antialias=True)])

palette = read_palette(PALETTE_PATH)
download_dataset(URL, PATH)
train_dataloader, test_dataloader = get_dataloaders(source_path=SOURCE,
                                                    target_path=TARGET,
                                                    batch_size=BATCH_SIZE,
                                                    train_size=TRAIN_SIZE,
                                                    palette=palette,
                                                    transforms=transforms)

model = DeepLabv3(CLASSES).to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=L_R, momentum=MOMENTUM)
criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

batches = len(train_dataloader)
max_iter = batches * EPOCHS
scheduler = CustomLRScheduler(optimizer, max_iter, L_R, POWER)

train(EPOCHS, model, optimizer, criterion, train_dataloader, test_dataloader, MODEL_NAME, DEVICE, scheduler)
