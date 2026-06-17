import torch


class ActivationMonitor:
    """
    Captura ativações intermediárias via forward hooks.
    Por padrão, registra hooks em todos os módulos folha (sem filhos),
    ou seja, captura a SAÍDA de cada Linear/ReLU/etc individualmente.
    Use layer_filter para restringir (ex: só pós-não-linearidade).
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.activations = {}
        self.hooks = []

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()
        return hook

    def _register_hooks(self, layer_filter=None):
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:  # só folhas
                continue
            if layer_filter is not None and not layer_filter(name, module):
                continue
            h = module.register_forward_hook(self._hook_fn(name))
            self.hooks.append(h)

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_activations(self, loader, layer_filter=None):
        self._register_hooks(layer_filter)
        device = next(self.model.parameters()).device
        log = []
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                self.activations = {}
                out = self.model(x)
                pred_labels = out.argmax(dim=1).detach().cpu()
                log.append({
                    'true_labels': y.detach().cpu(),
                    'pred_labels': pred_labels,
                    'activations': dict(self.activations),
                })
        self._remove_hooks()
        if was_training:
            self.model.train()
        return log




# uso, em paralelo ao que você já fez com GradMonitor:
# teacher_act_monitor = ActivationMonitor(teacher, config_prof)
# log_teacher_act = teacher_act_monitor.get_activations(train_loader)
# act_ordered_teacher = activation_ordered(log_teacher_act)