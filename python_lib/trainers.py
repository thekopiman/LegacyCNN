import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import os
from torch.autograd import Variable
import numpy as np


class BaseTrainer:
    def __init__(self, model, optimiser = "adam"):
        self.model = model

        if optimiser == "adam":
            self.optimiser = optim.Adam(
                self.model.parameters(), lr=0.0005,
            )
        else:
            self.optimiser = optim.SGD(self.model.parameters(), lr=0.0005, momentum=0.9)

    def _train(
        self,
        train_loader,
        loss_function=torch.nn.NLLLoss(),
        num_epochs=50,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        model = self.model.to(device)
        model.train()

        optimiser = self.optimiser

        for epoch in range(num_epochs):
            loss_history = []
            print(f"Epoch: {epoch}/{num_epochs-1}")
            print("-----------------------------")
            start = time.perf_counter()
            for batch_id, (data, label) in enumerate(train_loader):
                data = Variable(data.to(device))
                target = Variable(label.to(device))

                optimiser.zero_grad()
                preds = model(data)
                # print("preds: ",preds)
                # print("target: ", target)
                loss = loss_function(preds, target)
                loss.backward()
                loss_history.append(loss.data.item())
                optimiser.step()

            print(f"Loss avg: {sum(loss_history)/len(loss_history)}")

            print(f"Time : {time.perf_counter() - start}")
            print("=============================")

        self.model = model
        return model

    def _test(
        self,
        test_loader,
        loss_function=torch.nn.NLLLoss(),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        y_dim = 6,
    ) -> None:
        model = self.model.to(device)
        model.eval()
        confusion_matrix = np.zeros((y_dim, y_dim))

        test_loss = 0
        correct = 0

        start = time.perf_counter()
        for batch_id, (data, label) in enumerate(test_loader):
            # print("Batch_id: ", batch_id)
            data = data.to(device)
            target = label.to(device)

            # print(data.shape)
            output = model(data)
            test_loss += loss_function(output, target).data.item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            # print(type(pred), type(target))
            pred_ = pred.cpu().numpy()
            target_ = target.cpu().numpy()
            
            for i in range(pred_.shape[0]):
                confusion_matrix[pred_[i], target_[i]] += 1

        test_loss = test_loss
        test_loss /= len(test_loader)  # loss function already averages over batch size
        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), accuracy
            )
        )
        print(f"Time : {time.perf_counter() - start}")
        
        return confusion_matrix


    @classmethod
    def _debugTest(
        self,
        test_loader,
        loss_function=torch.nn.CrossEntropyLoss(),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        model1=None,
        model2=None,
    ):
        model1 = model1.to(device)
        model2 = model2.to(device)
        model1.eval()
        model2.eval()

        model1_mid = []
        model2_mid = []
        for batch_id, (data, label) in enumerate(test_loader):
            # print("Batch_id: ", batch_id)
            data = Variable(data.to(device), volatile=True)
            target = Variable(label.to(device))

            # print(data.shape)
            output1 = model1(data)
            model1_mid.append(model1.returnMidlayer())
            output2 = model2(data)
            model2_mid.append(model2.returnMidlayer())

        return model1_mid, model2_mid, output1, output2

    def save_model(
        self,
        file_path: str,
        model,
    ) -> None:
        """Save the model

        Args:
            file_path (str): Path dest
            model (nn, optional): Model. Defaults to None.
        """
        torch.save(model.state_dict(), file_path)