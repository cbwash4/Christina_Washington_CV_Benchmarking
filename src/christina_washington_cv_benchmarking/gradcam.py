
import torch
import numpy as np


class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._save_activation
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_index=None):

        self.model.eval()
        self.model.zero_grad()

        output = self.model(image_tensor)

        if hasattr(output, "logits"):
            output = output.logits

        if class_index is None:
            class_index = int(
                output.argmax(dim=1).item()
            )

        score = output[:, class_index]

        score.backward()

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * self.activations
        ).sum(dim=1)

        cam = torch.relu(cam)

        cam = cam[0].cpu().numpy()

        # Normalize to 0-1
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam, class_index

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()
