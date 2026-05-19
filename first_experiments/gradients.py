from torch.func import vmap, grad
from torch.func import functional_call
import torch
import torch.nn as nn
import h5py
from pruning_utils import save_results_to_h5, get_indexes_diff
from torch.utils.data import DataLoader


class GradMonitor():

    def __init__(self, model,config):
        self.model = model
        self.gradients = {}
        self.config = config
        self.criterion = config['criterion']  # e.g., torch.nn.CrossEntropyLoss()
        self.input_dim = config['input_dim']
        self.is_cnn = any(isinstance(m, nn.Conv2d) for m in self.model.modules())
    
    def get_gradients(self, dataloader, save_file=None, layers=None, nneurons=0):
        """
        layers: None = todas, str = uma camada, list[str] = várias
        ex: layers='layers.conv1.weight'
            layers=['layers.conv1.weight', 'layers.conv2.weight']
        """
        if isinstance(layers, str):
            layers = [layers]
        
        self.model.eval()

        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        device = next(self.model.parameters()).device  # pega device do modelo

        params = {name: p for name, p in params.items() if layers is None or name in layers}
        buffers = {name: b for name, b in buffers.items() if layers is None or name in layers}
        
        results = []

        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device) 


            # -------- predictions (no grad) --------
            with torch.no_grad():
                preds = self.model(x).argmax(dim=1).detach().cpu()  # batch inteiro

                with torch.no_grad():
                    preds = self.model(x).argmax(dim=1).detach().cpu()
                
                grads = {name: [] for name in params.keys()}  # acumula gradientes por camada
                for xi, yi in zip(x, y):
                    sample_grads = grad(self._loss_fn)(params, buffers, xi, yi)
                    for name, g in sample_grads.items():
                       grads[name].append(g.detach().clone().cpu())
                    torch.cuda.empty_cache()
                    #     logits = self.model(x)
                    #     preds = logits.argmax(dim=1)

                    # # -------- per-sample weight gradients (batched) --------
                    # grads = vmap(
                    #     grad(self._loss_fn),
                    #     in_dims=(None, None, 0, 0)
                    # )(params, buffers, x, y)

                # -------- store everything per batch --------
                batch_data = {
                    "true_labels": y.detach().cpu(),          # (B,)
                    "pred_labels": preds.detach().cpu(),      # (B,)
                    "grads": {
                        name: torch.stack(gs)                 # (B, ...)
                        for name, gs in grads.items()
                    }
                }

                if save_file is not None:
                    save_results_to_h5(batch_data,save_file, self.config['output_dim'])
                else:
                    results.append(batch_data)
                    torch.cuda.empty_cache()

        if save_file is None:
            return results

    def monte_carlo_gradients(self, dataloader, nsamples_per_class=1000):

        self.model.eval()

        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        device = next(self.model.parameters()).device  # pega device do modelo

        results = []
        all_predictions = []
        all_labels = []
        all_train = []

        for x, y in dataloader:
            x, y = x.to(device), y.to(device) 
            with torch.no_grad():
                logits = self.model(x)
                preds = logits.argmax(dim=1)

            all_predictions.append(preds.detach().cpu())
            all_labels.append(y.detach().cpu())
            all_train.append(x.detach().cpu())

        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_train = torch.cat(all_train, dim=0)

        idx_predictions = all_predictions.argsort()
        ord_predictions = all_predictions[idx_predictions]

        indexes_positions = get_indexes_diff(ord_predictions)
        min_class_size = min(
            indexes_positions[i+1] - indexes_positions[i]
            for i in range(len(indexes_positions)-1)
        )

        nsamples = min(nsamples_per_class, min_class_size)
        
        xsubset = []
        ysubset = []
        for i in range(len(indexes_positions)-1):
            s = indexes_positions[i]
            f = indexes_positions[i+1]
            idx = idx_predictions[s:s+nsamples]
            xsubset.append(all_train[idx])
            ysubset.append(all_labels[idx])

        return xsubset, ysubset 
    
    def monte_carlo_dataloader(self, dataloader, samples_per_class, output):
        device = next(self.model.parameters()).device  # pega device do modelo
        
        xsubset, ysubset = self.monte_carlo_gradients(dataloader, samples_per_class)
        xsubset = torch.cat(xsubset, dim=0).to(device)  # move para GPU
        ysubset = torch.cat(ysubset, dim=0).to(device)  # move para GPU
        
        dataset = torch.utils.data.TensorDataset(xsubset, ysubset)
        dtloader = DataLoader(dataset, batch_size=samples_per_class, shuffle=False)
        pos = [i*samples_per_class for i in range(output+1)] 
        return dtloader, pos

    # depois de selecionar as observacoes, pegar os gradientes e realizar os calculos
    # também preciso reescalar outra funcao para a partir de um par de informacao eu fazer isso
    def _loss_fn(self, params, buffers, x, y):
        """
        x: single sample (no batch dim)
        y: scalar
        """
        device = next(iter(params.values())).device
        if self.is_cnn:
            # x: (C, H, W) or (H, W)
            if x.dim() == 2:        # (H, W)
                x = x.unsqueeze(0)  # (1, H, W)
            x = x.unsqueeze(0)      # (1, C, H, W)
        else:
            # MLP: flatten everything
            x = x.view(-1).unsqueeze(0)  # (1, num_features)

        x = x.to(device)  # move para GPU
        y = y.unsqueeze(0).to(device)

        # y = y.unsqueeze(0)  # (1,)

        out = functional_call(self.model, (params, buffers), (x,))
        return self.criterion(out, y)


    def model_input_shape(self):
        """
        Returns expected input shape including batch dimension.
        - CNN: (1, channels, height, width), assumes grayscale if channels not given
        - MLP: (1, num_features)
        """
        input_dim = self.config['input_dim']  # e.g., (8, 8)

        # Detect if model is CNN
        is_cnn = any(isinstance(m, torch.nn.Conv2d) for m in self.model.modules())

        if is_cnn:
            # assume grayscale (1 channel)
            if len(input_dim) == 2:
                channels = 1
                height, width = input_dim
            else:
                raise ValueError(f"input_dim {input_dim} not compatible with CNN")
            return (1, channels, height, width)
        else:
            # MLP: flatten height*width into features
            if len(input_dim) == 2:
                num_features = input_dim[0] * input_dim[1]
            elif len(input_dim) == 1:
                num_features = input_dim[0]
            else:
                raise ValueError(f"input_dim {input_dim} not compatible with MLP")
            return (1, num_features)


