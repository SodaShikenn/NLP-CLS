from config import *
from utils import *
from model import *

if __name__ == '__main__':
    id2label, _ = get_label()

    train_dataset = Dataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=500, shuffle=True)

    dev_dataset = Dataset('dev')
    dev_loader = data.DataLoader(dev_dataset, batch_size=500, shuffle=True)

    model = TextCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    loss_fn = nn.CrossEntropyLoss()

    for e in range(EPOCH):
        for b, (input, mask, target) in enumerate(train_loader):

            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE) # Move target to the same device as model and input

            pred = model(input, mask)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % 50 != 0:
                continue

            y_pred = torch.argmax(pred, dim=1)
            report = evaluate(y_pred.cpu().data.numpy(), target.cpu().data.numpy(), output_dict=True) # Move target to CPU for evaluation

            with torch.no_grad():
                dev_input, dev_mask, dev_target = next(iter(dev_loader))

                dev_input = dev_input.to(DEVICE)
                dev_mask = dev_mask.to(DEVICE)
                dev_target = dev_target.to(DEVICE) # Move dev_target to the same device

                dev_pred = model(dev_input, dev_mask)
                dev_pred_ = torch.argmax(dev_pred, dim=1)
                dev_report = evaluate(dev_pred_.cpu().data.numpy(), dev_target.cpu().data.numpy(), output_dict=True) # Move dev_target to CPU for evaluation

            print(
                '>> epoch:', e,
                'batch:', b,
                'loss:', round(loss.item(), 5),
                'train_acc:', report['accuracy'],
                'dev_acc:', dev_report['accuracy'],
                'lr:', scheduler.get_last_lr()[0],
            )

        # Step the scheduler at the end of each epoch
        scheduler.step()

        if e % 10==0:
            torch.save(model, MODEL_DIR + f'model_{e}.pth')
