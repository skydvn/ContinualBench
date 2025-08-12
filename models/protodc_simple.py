# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


class ProtoDC(ContinualModel):
    """Continual learning via Prototype Set Condensation."""
    NAME = 'protodc'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight for prototype loss.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight for prototype alignment loss.')
        parser.add_argument('--lr_img', type=float, default=0.1,
                            help='Learning rate for synthetic images.')
        parser.add_argument('--proto_steps', type=int, default=10,
                            help='Number of steps for prototype optimization.')
        parser.add_argument('--proto_temp', type=float, default=0.1,
                            help='Temperature for prototype similarity.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size)
        self.proto_temp = args.proto_temp
        self.proto_steps = args.proto_steps

        # Get dataset properties
        self.num_classes = dataset.N_CLASSES_PER_TASK if dataset else 10
        self.input_shape = getattr(dataset, 'INPUT_SHAPE', (3, 32, 32))
        self.channel, self.im_size = self.input_shape[0], self.input_shape[1:]

        # Initialize prototypes storage
        self.class_prototypes = {}
        self.seen_classes = set()

    def extract_features(self, x):
        """Extract feature representations from the network."""
        # Assuming the backbone has a feature extraction method
        if hasattr(self.net, 'features'):
            return self.net.features(x)
        else:
            # Use penultimate layer features
            features = self.net(x, returnt='features')  # Modify based on your backbone
            return features

    def compute_prototype_loss(self, features, labels):
        """Compute prototypical network loss."""
        unique_labels = torch.unique(labels)
        proto_loss = 0

        for label in unique_labels:
            # Get features for this class
            class_mask = (labels == label)
            class_features = features[class_mask]

            if len(class_features) > 0:
                # Compute class prototype (centroid)
                prototype = class_features.mean(dim=0)

                # Store/update class prototype
                self.class_prototypes[label.item()] = prototype.detach()
                self.seen_classes.add(label.item())

                # Compute distance to prototype
                distances = torch.cdist(class_features.unsqueeze(0),
                                        prototype.unsqueeze(0).unsqueeze(0))
                proto_loss += distances.mean()

        return proto_loss / len(unique_labels) if len(unique_labels) > 0 else torch.tensor(0.0).to(features.device)

    def generate_synthetic_prototypes(self, current_labels):
        """Generate synthetic prototypical data for condensation."""
        device = next(self.net.parameters()).device

        # Initialize synthetic images for seen classes
        image_syn = []
        label_syn = []

        for class_id in self.seen_classes:
            # Create synthetic image for this class
            syn_img = torch.randn(size=(1, self.channel, self.im_size[0], self.im_size[1]),
                                  dtype=torch.float, requires_grad=True, device=device)
            image_syn.append(syn_img)

            # Create corresponding label
            syn_label = torch.tensor([class_id], dtype=torch.long, device=device)
            label_syn.append(syn_label)

        return image_syn, label_syn

    def optimize_prototypes(self, image_syn, label_syn, target_features):
        """Optimize synthetic prototypes to match target features."""
        if not image_syn:
            return

        # Optimizer for synthetic images
        optimizer_img = SGD([{'params': img, 'lr': self.args.lr_img} for img in image_syn],
                            momentum=0.5)

        criterion_proto_align = nn.MSELoss()

        for step in range(self.proto_steps):
            optimizer_img.zero_grad()
            total_align_loss = 0

            for i, (syn_img, syn_label) in enumerate(zip(image_syn, label_syn)):
                # Get features from synthetic image
                syn_features = self.extract_features(syn_img)

                # Get target prototype for this class
                class_id = syn_label.item()
                if class_id in self.class_prototypes:
                    target_proto = self.class_prototypes[class_id]

                    # Align synthetic features with target prototype
                    align_loss = criterion_proto_align(syn_features.mean(dim=0), target_proto)
                    total_align_loss += align_loss

            if total_align_loss > 0:
                total_align_loss.backward()
                optimizer_img.step()

    def compute_buffer_alignment_loss(self, current_features, current_labels):
        """Compute alignment loss between current and buffered prototypes."""
        if self.buffer.is_empty():
            return torch.tensor(0.0).to(current_features.device)

        # Get buffered data
        buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
            min(self.args.minibatch_size, len(self.buffer)),
            transform=self.transform,
            device=self.device
        )

        # Extract features from buffered data
        buf_features = self.extract_features(buf_inputs)

        criterion_proto_align = nn.MSELoss()
        align_loss = 0
        count = 0

        # Align prototypes of common classes
        current_unique = torch.unique(current_labels)
        buf_unique = torch.unique(buf_labels)
        common_classes = set(current_unique.cpu().numpy()) & set(buf_unique.cpu().numpy())

        for class_id in common_classes:
            # Current class prototype
            current_mask = (current_labels == class_id)
            if current_mask.sum() > 0:
                current_proto = current_features[current_mask].mean(dim=0)

                # Buffered class prototype
                buf_mask = (buf_labels == class_id)
                if buf_mask.sum() > 0:
                    buf_proto = buf_features[buf_mask].mean(dim=0)

                    # Alignment loss
                    align_loss += criterion_proto_align(current_proto, buf_proto)
                    count += 1

        return align_loss / count if count > 0 else torch.tensor(0.0).to(current_features.device)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """Main training step with prototype condensation."""

        # Prototype + Network Optimizer
        self.opt.zero_grad()

        # Forward pass
        outputs = self.net(inputs)
        features = self.extract_features(inputs)

        # Vanilla Loss (standard classification)
        vanilla_loss = self.loss(outputs, labels)

        # ProtoNet Loss (prototypical network loss)
        proto_loss = self.compute_prototype_loss(features, labels)

        # Combined loss
        loss = vanilla_loss + self.args.alpha * proto_loss

        # Prototype alignment loss with buffer
        if not self.buffer.is_empty():
            # Compute alignment loss between current and stored prototypes
            loss_pa = self.args.beta * self.compute_buffer_alignment_loss(features, labels)
            loss += loss_pa

            # Generate and optimize synthetic prototypes for condensation
            image_syn, label_syn = self.generate_synthetic_prototypes(labels)
            if image_syn:  # Only if we have synthetic prototypes
                self.optimize_prototypes(image_syn, label_syn, features)

        # Backward pass and optimization
        loss.backward()
        self.opt.step()

        # Add prototypical exemplars to buffer
        # Store examples with their features/logits for future prototype alignment
        with torch.no_grad():
            # Select representative examples (could be improved with more sophisticated selection)
            if len(not_aug_inputs) > 0:
                self.buffer.add_data(
                    examples=not_aug_inputs,
                    labels=labels,
                    logits=outputs.data
                )

        return loss.item()

    def forward(self, x):
        """Forward pass through the network."""
        return self.net(x)

    def get_prototype_similarity(self, features, class_id):
        """Get similarity to stored prototype for a specific class."""
        if class_id not in self.class_prototypes:
            return torch.tensor(0.0)

        prototype = self.class_prototypes[class_id]
        similarity = F.cosine_similarity(features, prototype.unsqueeze(0), dim=1)
        return similarity.mean()

    def update_prototypes(self, features, labels):
        """Update class prototypes with new data using moving average."""
        unique_labels = torch.unique(labels)
        momentum = 0.9  # Momentum for prototype updates

        for label in unique_labels:
            class_mask = (labels == label)
            class_features = features[class_mask]

            if len(class_features) > 0:
                new_prototype = class_features.mean(dim=0).detach()
                label_item = label.item()

                if label_item in self.class_prototypes:
                    # Update existing prototype with momentum
                    self.class_prototypes[label_item] = (
                            momentum * self.class_prototypes[label_item] +
                            (1 - momentum) * new_prototype
                    )
                else:
                    # Initialize new prototype
                    self.class_prototypes[label_item] = new_prototype

                self.seen_classes.add(label_item)
