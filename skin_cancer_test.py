# Bla bla bkla

#Mercedes es mi diosa
from torchvision import transforms, datasets
import torchvision
import torch
import time
from matplotlib import pyplot
import numpy

device='cuda'

if __name__ == "__main__":

    transforms_train=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    transforms_test=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    train_dir = "/home/224A1087sergio/Skin_Cancer_demo/train"
    test_dir = "/home/224A1087sergio/Skin_Cancer_demo/test"

    train_dataset = datasets.ImageFolder(train_dir, transforms_train)
    test_dataset = datasets.ImageFolder(test_dir, transforms_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = torchvision.models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    print(num_features)

    model.fc = torch.nn.Linear(512, 2)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.00001)

    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    num_epochs = 10
    start_time = time.time()

    for epoch in range(num_epochs):
        print("Epoch {} running".format(epoch))
        """ Training phase """
        model.train()
        running_loss = 0.
        running_corrects = 0

        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset)*100.

        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_acc)
        # Print progress
        print('[Train] Loss: {:.4f} Acc: {:.2f}%'.format(epoch_loss, epoch_acc))

        model.eval()
        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0

        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects / len(test_dataset)*100.

        test_loss.append(epoch_loss)
        test_accuracy.append(epoch_acc)

        print('[Test] Loss: {:.4f} Acc: {:.2f}%'.format(epoch_loss, epoch_acc))

    pyplot.clf()
    pyplot.plot(test_loss, label='TEST')
    pyplot.plot(train_loss, label='TRAIN')
    pyplot.legend()
    pyplot.title("Loss")
    pyplot.savefig("Loss.jpg", format='jpg')
    pyplot.show()

    pyplot.clf()
    pyplot.plot(test_accuracy, label='TEST')
    pyplot.plot(train_accuracy, label='TRAIN')
    pyplot.legend()
    pyplot.title("Accuracy")
    pyplot.savefig("Accuracy.jpg", format='jpg')
    pyplot.show()
