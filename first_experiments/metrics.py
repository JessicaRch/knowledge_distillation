import torch
class Evaluator:
    def __init__(self, model):
        self.model = model

    def accuracy(self, dataloader):
        self.model.eval()

        correct = 0
        total = 0
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)

                logits = self.model(x)
                preds = logits.argmax(dim=1)

                total += y.size(0)
                correct += (preds == y).sum().item()

        return correct / total
