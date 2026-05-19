import torch
class Trainer:
    
    def __init__(self, model, config):
        self.model = model
        self.optimizer = config['optimizer'](model.parameters(), lr=config['lr'])
        self.criterion = config['criterion']
        self.device = next(model.parameters()).device

    def train_epoch(self, dataloader):
        
        self.model.train()
        running_loss = 0.0

        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * data.size(0)

        return running_loss / len(dataloader.dataset)

    def train_student(self, teacher, dataloader, T=1.0, c=0.5):

        self.model.train() # student is being trained
        teacher.eval()  # Teacher is fixed
        running_loss = 0.0

        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(data)  # Get teacher's output
            
            student_logits = self.model(data)  # Get student's output
            teacher_soft = torch.nn.functional.softmax(teacher_logits.detach() / T, dim=-1)
            student_soft = torch.nn.functional.log_softmax(student_logits / T, dim=-1)
            
            L_soft = torch.nn.functional.kl_div(student_soft, teacher_soft, reduction='batchmean')  # KL divergence
            L_hard = self.criterion(student_logits, target)  # CE normal
            loss = (1-c) * T**2 *  L_soft + c * L_hard  # Combine losses
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * data.size(0)

        return running_loss / len(dataloader.dataset)

if __name__ == '__main__':
    # main.py
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from pruning_utils import low_activity_neurons, create_adjacency_list, chose_neurons_to_prune

    from mnist_dataset import MNISTDataset

    from metrics import Evaluator
    from pruner import IterativePruner, Pruner
    import torch
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from gradients import GradMonitor
    from DSU import getComponents
    from model import CNN
    from training import Trainer
    from digits_dataset import DigitsDataset

    digits_dataset = DigitsDataset(cnn=True)
    train_loader, test_loader = digits_dataset.get_loaders(batch_size=32)

    x,y = next(iter(train_loader))
    input_dim = x.shape[2:]
    output_dim = 10

    config = {
        'input_dim': input_dim,  # (height, width)
        'conv': [16, 32],  
        'fc' : [64],
        'conv_kernel' : [3,3, 3], 
        'conv_stride': [1, 1, 1],
        'conv_padding': [1, 1, 1],
        'min_neurons_per_conv_layer': [4, 4],
        'min_neurons_per_fc_layer': [4],
        'max_pool_kernel' : [2,2],
        'max_pool_stride' : [2,2],
        'output_dim': output_dim,
        'criterion': torch.nn.CrossEntropyLoss(),
        'optimizer': torch.optim.Adam,
        'lr': 0.001
    }

    cnn_model = CNN(config)
    trainer = Trainer(cnn_model, config)
    for _ in range(20):
        trainer.train_epoch(train_loader)

    evaluator = Evaluator(cnn_model)
    train_acc, test_acc = evaluator.accuracy(train_loader), evaluator.accuracy(test_loader)
    grad_monitor = GradMonitor(cnn_model, config, is_cnn=True)
    print("Initial train acc:", train_acc, "test acc:", test_acc)
    pruner = IterativePruner(cnn_model, config, train_loader, test_loader, max_iters=2,
                            threshold_low_act=0.09, threshold_sim=0.8, use_cuda=False, eps=0.15)
    pruner.run(retrain_epochs=10)