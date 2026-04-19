import torch

from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def train_and_val(model,
                  train_loader,
                  val_loader,
                  optimizer,
                  criterion,
                  device="cpu",
                  epochs=30):
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(epochs):
        running_train_loss = 0
        correct_train = 0
        total_train = 0
        model.train()
        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            preds = output.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_train_loss / len(train_loader)
        train_acc = correct_train / total_train

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        running_val_loss = 0
        correct_val = 0
        total_val = 0
        last_epoch_preds = []
        last_epoch_labels = []
        model.eval()
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader):
                imgs, labels = imgs.to(device), labels.to(device)

                output = model(imgs)
                loss = criterion(output, labels)

                running_val_loss += loss.item()
                preds = output.agrmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

            val_loss = running_val_loss / len(val_loader)
            val_acc = correct_val / total_val

            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if epoch == epochs - 1:
                last_epoch_preds.append(preds.numpy().cpu())
                last_epoch_labels.append(labels.numpy().cpu())

            if best_val_loss > val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'valid_loss': val_loss,
                    }, "model1.pth")
                
        print(f"epoch: {epoch+1} | train_loss: {train_loss} | val_loss: {val_loss}")

        cm = confusion_matrix(last_epoch_labels, last_epoch_preds)

    return model, train_losses, train_accuracies, val_losses, val_accuracies, cm



