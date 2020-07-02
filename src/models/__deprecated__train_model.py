import torch
from tqdm import tqdm

def train_net(net, opt, loss_fn, n_epoch, 
        train_loader, val_loader, device='cpu', ax=None):
    net.to(device)
    train_losses, train_acces = [], []
    val_losses, val_acces = [], []
    for epoch in range(n_epoch):
        train_loss, train_acc = train_epoch(epoch, 
                net, opt, loss_fn, train_loader, device)
        train_losses.append(train_loss)
        train_acces.append(train_acc)
        val_loss, val_acc = val_epoch(epoch,
                net, loss_fn, val_loader, device)
        val_losses.append(val_loss)
        val_acces.append(val_acc)
        print('t: {:.2f}, v: {:.2f}'.format(train_acc, val_acc))
    ax_x = list(range(n_epoch))
    ax.plot(ax_x, train_losses, label='t')
    ax.plot(ax_x, val_losses, label='v')
    ax.legend()
    return train_losses, train_acces, val_losses, val_acces

def train_epoch(epoch, net, opt, loss_fn, train_loader, device='cpu'):
    net.train()
    n_samples, run_loss, n_correct = 0, 0.0, 0
    for batch in tqdm(train_loader, total=len(train_loader), desc=str(epoch)):
        n, l, c = train_batch(net, opt, loss_fn, batch, device)
        n_samples += n
        run_loss += n * l
        n_correct += c
    loss, acc = (run_loss / n_samples), (n_correct / n_samples)
    return loss, acc

def train_batch(net, opt, loss_fn, batch, device='cpu'):
    X_batch, y_batch = batch.values()
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    h_batch = net(X_batch)
    _, y_pred_batch = h_batch.max(1)
    loss = loss_fn(h_batch, y_batch)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return len(y_batch), loss.item(),\
            (y_pred_batch == y_batch).float().sum().item()

# 코드 절약 및 수정 용이성을 위해 run_epoch, run_batch로 바꾸는 것도 괜찮아 보임
def val_epoch(epoch, net, loss_fn, val_loader, device='cpu'):
    net.eval()
    n_samples, run_loss, n_correct = 0, 0.0, 0
    for batch in val_loader:
        n, l, c = val_batch(net, loss_fn, batch, device)
        n_samples += n
        run_loss += n * l
        n_correct += c
    loss, acc = (run_loss / n_samples), (n_correct / n_samples)
    return loss, acc

def val_batch(net, loss_fn, batch, device='cpu'):
    X_batch, y_batch = batch.values()
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    with torch.no_grad():
        h_batch = net(X_batch)
        _, y_pred_batch = h_batch.max(1)
        loss = loss_fn(h_batch, y_batch)
        _, y_pred_batch = h_batch.max(1)
        return len(y_batch), loss.item(),\
                (y_pred_batch == y_batch).float().sum().item()