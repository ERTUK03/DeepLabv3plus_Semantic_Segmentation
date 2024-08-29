import torch

def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               criterion: torch.nn.Module,
               device: torch.device,
               scheduler: torch.optim.lr_scheduler):

    running_loss = 0.
    last_loss = 0.

    model.train()
    for i, (images, masks) in enumerate(train_dataloader):
        images = images.to(device)
        masks = masks.clone().detach().to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        scheduler.step()
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss /10
            print(f'batch {i+1} loss: {last_loss}')
            running_loss = 0.

    return last_loss

def test_step(model: torch.nn.Module,
              test_dataloader: torch.utils.data.DataLoader,
              criterion: torch.nn.Module,
              device: torch.device):

    running_vloss = 0.
    model.eval()

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_dataloader):
            images = images.to(device)
            masks = masks.clone().detach().to(device)
            outputs = model(images)
            vloss = criterion(outputs, masks)
            running_vloss += vloss.item()
    avg_loss = running_vloss / (i + 1)
    return avg_loss

def train(epochs: int,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          model_path: str,
          device: torch.device,
          scheduler: torch.optim.lr_scheduler):

    best_vloss = 1000000

    for epoch in range(1,epochs+1):
        print(f'EPOCH {epoch} LR {scheduler.get_lr()[0]}')
        avg_loss = train_step(model, train_dataloader, optimizer, criterion, device, scheduler)
        avg_vloss = test_step(model, test_dataloader, criterion, device)
        print(f'LOSS train {avg_loss} test {avg_vloss}')
        if avg_vloss<best_vloss:
            best_vloss = avg_vloss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_vloss
                }, f'{model_path}_{epoch}.pth')
