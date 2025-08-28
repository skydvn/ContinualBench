# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from argparse import ArgumentParser

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
        parser.add_argument('--proto_steps', type=int, default=1,
                            help='Number of steps for prototype optimization.')
        parser.add_argument('--proto_temp', type=float, default=0.1,
                            help='Temperature for prototype similarity.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)


        # Define Buffer
        self.buffer = Buffer(self.args.buffer_size)
        # FIXME This buffer has size equal to number of classes,
        # FIXME When new prototypical exemplar according to one class is added, the last is replaced.
        self.buf_labels = {}
        self.buf_syn_img = {}

        self.proto_temp = args.proto_temp
        self.proto_steps = args.proto_steps

        # Get dataset properties
        self.num_classes = dataset.N_CLASSES_PER_TASK if dataset else 10
        self.input_shape = getattr(dataset, 'INPUT_SHAPE', (3, 32, 32))
        self.channel, self.im_size = self.input_shape[0], self.input_shape[1:]

        # Initialize prototypes storage
        self.class_prototypes = {}      # This is stored every round and reset in the next round
        self.class_exemplars = {}
        self.seen_classes = set()

        # Initialize learnable synthetic prototypical exemplars:
        self.image_syn = []
        self.label_syn = []

        # TODO Initialize learning epochs/iters for model / synthetic data
        self.n_epoch_model = self.args.n_epochs // 2
        self.n_epoch_data = self.args.n_epochs - self.n_epoch_model

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
        """
        Compute prototypical network loss following the formula:

        For k in {1, ..., N_C} do:
            For (x, y) in Q_k do:
                J ← J + (1/(N_C × N_Q)) × [d(f_φ(x), c_k) + log(∑_{k'} exp(-d(f_φ(x), c_k')))]

        Args:
            features: Feature representations f_φ(x) of shape (batch_size, feature_dim)
            labels: Ground truth labels of shape (batch_size,)

        Returns:
            Prototypical network loss (scalar tensor)
        """
        device = features.device
        unique_labels = torch.unique(labels)
        N_C = len(unique_labels)  # Number of classes in current batch

        if N_C == 0:
            return torch.tensor(0.0, device=device)

        # Step 1: Compute prototypes c_k for each class k
        prototypes = {}
        class_counts = {}

        for label in unique_labels:
            class_mask = (labels == label)
            class_features = features[class_mask]  # All features for class k

            if len(class_features) > 0:
                # c_k = mean of all features for class k
                prototypes[label.item()] = class_features.mean(dim=0)
                class_counts[label.item()] = len(class_features)

                # Update stored prototypes for future use
                self.class_prototypes[label.item()] = prototypes[label.item()].detach()
                self.seen_classes.add(label.item())

        if len(prototypes) == 0:
            return torch.tensor(0.0, device=device)

        # Step 2: Compute prototypical network loss
        total_loss = 0.0
        N_Q = 0  # Total number of query samples

        # For each class k in {1, ..., N_C}
        for k in prototypes.keys():
            c_k = prototypes[k]  # Prototype for class k

            # Get all query samples (x, y) in Q_k (where y = k)
            query_mask = (labels == k)
            query_features = features[query_mask]  # f_φ(x) for all x where y = k

            # For each query sample (x, y) in Q_k
            for query_feature in query_features:
                # Compute distances d(f_φ(x), c_k') for all k'
                distances = []

                for k_prime in prototypes.keys():
                    c_k_prime = prototypes[k_prime]
                    # Squared Euclidean distance: d(f_φ(x), c_k')
                    dist = torch.sum((query_feature - c_k_prime) ** 2)
                    distances.append(dist)

                distances = torch.stack(distances)  # Shape: (num_classes,)

                # Find distance to correct prototype c_k
                class_keys = list(prototypes.keys())
                true_class_idx = class_keys.index(k)
                d_true = distances[true_class_idx]  # d(f_φ(x), c_k)

                # Compute log-sum-exp: log(∑_{k'} exp(-d(f_φ(x), c_k'))) where k' ≠ k
                # Exclude the true class distance from the sum
                distances_excluding_true = torch.cat([
                    distances[:true_class_idx],
                    distances[true_class_idx + 1:]
                ])
                log_sum_exp = torch.logsumexp(-distances_excluding_true, dim=0)

                # Compute loss for this sample: d(f_φ(x), c_k) + log(∑_{k'} exp(-d(f_φ(x), c_k')))
                sample_loss = d_true + log_sum_exp
                total_loss += sample_loss
                N_Q += 1

        # Normalize by total number of samples: 1/(N_C × N_Q)
        if N_Q > 0:
            normalized_loss = total_loss / (N_C * N_Q)
            return normalized_loss
        else:
            return torch.tensor(0.0, device=device)

    # Alternative vectorized implementation for better efficiency
    def compute_prototype_loss_vectorized(self, features, labels):
        """
        Vectorized version of prototypical network loss computation.
        More efficient for larger batches.
        """
        device = features.device
        unique_labels = torch.unique(labels)
        N_C = len(unique_labels)

        if N_C == 0:
            return torch.tensor(0.0, device=device)

        # Compute prototypes for each class
        prototypes = []
        prototype_labels = []

        for label in unique_labels:
            class_mask = (labels == label)
            class_features = features[class_mask]

            if len(class_features) > 0:
                prototype = class_features.mean(dim=0)
                prototypes.append(prototype)
                prototype_labels.append(label)

                # Update stored prototypes for future use
                self.class_prototypes[label.item()] = prototypes[label.item()].detach()
                self.seen_classes.add(label.item())

        if len(prototypes) == 0:
            return torch.tensor(0.0, device=device)

        # Stack prototypes: (num_classes, feature_dim)
        prototypes = torch.stack(prototypes)
        prototype_labels = torch.tensor(prototype_labels, device=device)

        # Compute pairwise squared distances: (batch_size, num_classes)
        # ||f_φ(x) - c_k||²
        distances = torch.cdist(features.unsqueeze(1), prototypes.unsqueeze(0)).squeeze(1) ** 2

        # For each sample, find the index of its true class prototype
        true_class_indices = []
        for i, label in enumerate(labels):
            try:
                idx = (prototype_labels == label).nonzero(as_tuple=True)[0][0]
                true_class_indices.append(idx)
            except IndexError:
                # Handle case where prototype for this label doesn't exist
                true_class_indices.append(0)  # Default to first prototype

        true_class_indices = torch.tensor(true_class_indices, device=device)

        # Get distances to true class prototypes: d(f_φ(x), c_k)
        true_distances = distances[torch.arange(len(labels)), true_class_indices]

        # Compute log-sum-exp over all prototypes EXCEPT true class: log(∑_{k'≠k} exp(-d(f_φ(x), c_k')))
        # Create mask to exclude true class distances
        batch_indices = torch.arange(len(labels), device=device)

        # Set true class distances to -inf so they don't contribute to logsumexp
        masked_distances = distances.clone()
        masked_distances[batch_indices, true_class_indices] = float('-inf')

        log_sum_exp = torch.logsumexp(-masked_distances, dim=1)

        # Compute loss for each sample: d(f_φ(x), c_k) + log(∑_{k'} exp(-d(f_φ(x), c_k')))
        sample_losses = true_distances + log_sum_exp

        # Average over all samples: 1/(N_C × N_Q)
        N_Q = len(labels)
        normalized_loss = sample_losses.mean() / N_C

        return normalized_loss

    def generate_synthetic_prototypes(self):
        """Generate synthetic prototypical data for condensation."""
        device = next(self.net.parameters()).device

        # FIXME At beginning of each task (task_idx + epoch = 0), init new learnable prototypes
        # FIXME By using self.image_syn, we can store the state + reset whenever it required
        if self.task_iteration == 0 and self._epoch_iteration == 0:
            # Initialize synthetic images for seen classes
            self.image_syn = []
            self.label_syn = []

            for class_id in self.seen_classes:
                # Create synthetic image for this class
                syn_img = torch.randn(size=(1, self.channel, self.im_size[0], self.im_size[1]),
                                      dtype=torch.float, requires_grad=True, device=device)
                self.image_syn.append(syn_img)

                # Create corresponding label
                syn_label = torch.tensor([class_id], dtype=torch.long, device=device)
                self.label_syn.append(syn_label)
        else:
            # FIXME Temporarily I will test code flow by just passing the condition.
            pass
            # # Initialize synthetic images for seen classes
            # image_syn = []
            # label_syn = []
            #
            # for class_id in self.seen_classes:
            #     # Reuse the synthetic data
            #     syn_img = self.buf_syn_img[class_id]
            #     image_syn.append(syn_img)
            #
            #     # Create corresponding label
            #     syn_label = self.buf_labels[class_id]
            #     label_syn.append(syn_label)


    def optimize_proto_exemplar(self, image_syn, label_syn, target_features=None):
        """
        Optimize synthetic prototypes to match target features.

        This function optimizes each synthetic exemplar image_syn[i] to produce features
        that match the stored prototype for class i.

        Args:
            image_syn: List of synthetic images [image_syn[0], image_syn[1], ..., image_syn[num_classes-1]]
                       where image_syn[i] is the synthetic exemplar for class i
            label_syn: List of labels [0, 1, 2, ..., num_classes-1]
            target_features: Optional target features (not used in this implementation)
        """
        if not image_syn:
            return

        criterion_proto_align = nn.MSELoss()
        # Optimize each synthetic exemplar individually
        for i, (syn_img, syn_label) in enumerate(zip(image_syn, label_syn)):
            class_id = syn_label.item()

            # Skip if we haven't seen this class yet (no prototype to align to)
            if class_id not in self.class_prototypes:
                continue

            # Create optimizer for this specific synthetic image
            optimizer_img = SGD([syn_img], lr=self.args.lr_img, momentum=0.5)
            target_proto = self.class_prototypes[class_id]

            # Optimize this synthetic exemplar over multiple steps
            for step in range(self.proto_steps):
                optimizer_img.zero_grad()
                total_loss = 0

                # Forward pass: get features from synthetic image
                # syn_img has shape (1, C, H, W) - single exemplar image
                syn_features = self.extract_features(syn_img)

                # syn_features has shape (1, feature_dim) - features from the single exemplar
                # The exemplar itself IS the prototype, so we squeeze the batch dimension
                syn_proto = syn_features.squeeze(0)  # Remove batch dimension: (feature_dim,)

                # Ensure target prototype has same shape
                if target_proto.dim() != syn_proto.dim():
                    if target_proto.dim() > 1:
                        target_proto = target_proto.view(-1)
                    if syn_proto.dim() > 1:
                        syn_proto = syn_proto.view(-1)

                # TODO Align synthetic prototype with stored / target prototype
                align_loss = criterion_proto_align(syn_proto, target_proto)
                # keep_loss = criterion_proto_align(syn_proto, buff_proto)
                total_loss = align_loss

                # Backward pass and optimization step
                total_loss.backward()
                optimizer_img.step()

                # Optional: Add some constraints to keep synthetic images realistic
                with torch.no_grad():
                    # Clamp pixel values to reasonable range (e.g., [0, 1] or [-1, 1])
                    syn_img.clamp_(-2.0, 2.0)  # Adjust range based on your data normalization

        print(f"average proto loss: {total_loss.item()}")

        # Update buffer with optimized synthetic exemplars
        self._add_synthetic_to_buffer(image_syn, label_syn)

    def _add_synthetic_to_buffer(self, image_syn, label_syn):
        """
        Add optimized synthetic exemplars to the buffer.

        Args:
            image_syn: List of optimized synthetic images
            label_syn: Corresponding labels
        """
        with torch.no_grad():
            for syn_img, syn_label in zip(image_syn, label_syn):
                class_id = syn_label.item()
                # FIXME Add synthetic exemplar to buffer
                # FIXME buffer size 1xCxHxW
                self.buf_syn_img[class_id] = syn_img.to(self.device)
                self.buf_labels[class_id] = syn_label.to(self.device)


    def compute_buffer_alignment_loss(self, current_features, current_labels):
        """
        Compute alignment loss between current and buffered prototypes.

        Assumes buf_inputs are already unique exemplars - each buf_inputs[i]
        corresponds to a unique class in buf_labels[i].

        Args:
            current_features: Features from current batch, shape (batch_size, feature_dim)
            current_labels: Labels from current batch, shape (batch_size,)

        Returns:
            Alignment loss (scalar tensor)
        """
        # Check if buffer has any data
        if not hasattr(self, 'buf_labels') or len(self.buf_labels) == 0:
            return torch.tensor(0.0, device=current_features.device)

        criterion_proto_align = nn.MSELoss()
        align_loss = 0
        count = 0

        """
            Args:
            - self.buf_labels: dict {key: values}
            - self.buf_syn_img: dict {key: values}

        """
        for class_id in self.seen_classes:
            # Current class prototype (average of current samples for this class)
            current_mask = (current_labels == class_id)
            if current_mask.sum() > 0:
                current_proto = current_features[current_mask].mean(dim=0)  # 1xD

                if not self.buf_syn_img[class_id] is None:
                    _ , buf_proto = self.net(self.buf_syn_img[class_id], returnt='both') # Single exemplar feature

                    # Ensure prototypes have same shape
                    if current_proto.shape != buf_proto.shape:
                        current_proto = current_proto.view(-1)
                        buf_proto = buf_proto.view(-1)

                    # Alignment loss between current prototype and buffered exemplar
                    align_loss += criterion_proto_align(current_proto, buf_proto)
                    count += 1

        return align_loss / count if count > 0 else torch.tensor(0.0, device=current_features.device)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """Main training step with prototype condensation.
            - First n_epoch // 2 steps:
                Train Model
            - Last n_epoch // 2 steps:
                Free Model + Train Data
        """
        if epoch <= self.n_epoch_model:
            # Prototype + Network Optimizer
            self.opt.zero_grad()

            # Forward pass
            outputs, features = self.net(inputs, returnt = 'both')

            # Vanilla Loss (standard classification)
            vanilla_loss = self.loss(outputs, labels)

            # ProtoNet Loss (prototypical network loss)
            proto_loss = self.compute_prototype_loss(features, labels)

            # Combined loss
            loss = vanilla_loss + self.args.alpha * proto_loss

            # Prototype alignment loss with buffer
            # Check if buffer has any data
            if not hasattr(self, 'buf_labels') or len(self.buf_labels) == 0:
                # Compute alignment loss between current and stored prototypes
                # FIXME Need to store past inputs to support compute_buffer_alignment_loss
                loss_pa = self.args.beta * self.compute_buffer_alignment_loss(features, labels)
                loss += loss_pa

            # Backward pass and optimization
            loss.backward()
            self.opt.step()

            return loss.item()

        # Generate and optimize synthetic prototypes for condensation
        # FIXME Need flag to control the training (the task needed to be looped 2 times)
        # FIXME 1st for the model training // 2nd for the exemplar training

        # FIXME At beginning of each task (task_idx + epoch = 0), init new learnable prototypes
        # FIXME As this observe is set in the loop over dataset,
        # FIXME the buffer should only save 1 data according to each class.
        else:
            self.generate_synthetic_prototypes()
            if self.image_syn:  # Only if we have synthetic prototypes
                self.optimize_proto_exemplar(self.image_syn, self.label_syn, features)

            return 0

    def forward(self, x):
        """Forward pass through the network."""
        return self.net(x)
