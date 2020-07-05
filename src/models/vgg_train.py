from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.utils import convert_tensor

def train_net(net, opt, loss_fn, val_metrics, train_loader, val_loader, device):
    net.to(device)
    def prepare_batch(batch, device, non_blocking=False):
        x, y = batch.values()
        return (
                convert_tensor(x, device=device, non_blocking=non_blocking),
                convert_tensor(y, device=device, non_blocking=non_blocking),
        )
    def output_transform(x, y, y_pred, loss):
        return (y_pred.max(1)[1], y)
    trainer = create_supervised_trainer(net, opt, loss_fn, device,
            prepare_batch=prepare_batch, output_transform=output_transform)
    evaluator = create_supervised_evaluator(net, val_metrics, device,
            prepare_batch=prepare_batch)
    s = '{}: {:.2f} '
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        print('Epoch {}'.format(trainer.state.epoch))
        message = 'Train - '
        for m in val_metrics.keys():
            message += s.format(m, evaluator.state.metrics[m])
        print(message)
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        message = 'Val   - '
        for m in val_metrics.keys():
            message += s.format(m, evaluator.state.metrics[m])
        print(message)
    return trainer