# Copyright 2025-present, Minh-Duong Nguyen
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from argparse import ArgumentParser

from models.protocore_utils.proto_utils import AugmentationMixer
from models.protocore_utils.protocore_loss import euclidean_dist, HyperbolicContrastiveLoss
from models.protocore_utils.proto_visualize import Visualizer, HyperbolicVisualizer
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer

import os

class ProtoDC(ContinualModel):
    """Continual learning via Prototype Set Condensation."""
    NAME = 'protodc'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True, default = 0.1,
                            help='Penalty weight for prototype loss.')
        parser.add_argument('--beta', type=float, required=True, default = 0.1,
                            help='Penalty weight for prototype alignment loss.')
        parser.add_argument('--lr_img', type=float, default=0.5,
                            help='Learning rate for synthetic images.')
        parser.add_argument('--proto_steps', type=int, default=1,
                            help='Number of steps for prototype optimization.')
        parser.add_argument('--augment_flag', type=int, default=0,
                            help='Flag for augmentation.')
        parser.add_argument('--proto_temp', type=float, default=0.1,
                            help='Temperature for prototype similarity.')
        parser.add_argument('--patience', type=int, default=10,
                            help='Number of epochs to wait for improvement.')
        parser.add_argument('--spc', type=int, default=1,
                            help='Number of sample per class.')
        parser.add_argument('--min_delta', type=float, default=1e-4,
                            help='Minimum improvement to be considered as progress.')
        parser.add_argument('--proto_method', type=str, default='hyperbolic',
                            help='contrastive, prototypical, hyperbolic.')
        return parser


    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        note = "syn_proto"
        # self.visualizer = Visualizer(save_dir = f"./tsne/{note}")
        self.visualizer = HyperbolicVisualizer(save_dir= f"./tsne/{note}")

        # Define Buffer
        self.buffer = Buffer(self.args.buffer_size)
        # FIXME This buffer has size equal to number of classes,
        # FIXME When new prototypical exemplar according to one class is added, the last is replaced.
        self.buf_labels = {}
        self.buf_syn_img = {}

        self.proto_temp = args.proto_temp
        self.proto_steps = args.proto_steps
        self.spc = args.spc

        self.augmentation = AugmentationMixer(device="cuda")

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

        self.hyperbolic_cl = HyperbolicContrastiveLoss()

        # TODO Initialize learning epochs/iters for model / synthetic data
        self.n_epoch_model = self.args.n_epochs // 2
        self.n_epoch_data = self.args.n_epochs - self.n_epoch_model

        if not hasattr(self, "best_loss"):
            print("Best Loss Init")
            self.best_loss = float("inf")
            self.patience_counter = 0

            save_model_dir = f"checkpoints/best_model"
            os.makedirs(save_model_dir, exist_ok=True)  # Create folder if it does not exist
            self.best_model_path = os.path.join(save_model_dir, "best_model.pt")

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

    def prototypical_loss(self, input, target, n_query):
        """
        Compute prototypical loss and accuracy using fixed n_query per class.

        Args:
            input (Tensor): Model output of shape [N, D]
            target (Tensor): Ground truth labels of shape [N]
            n_query (int): Number of query samples per class

        Returns:
            Tuple[Tensor, Tensor]: loss and accuracy
        """
        classes = torch.unique(target)
        n_classes = len(classes)

        support_idxs = []
        query_idxs = []

        for cls in classes:
            cls_idxs = (target == cls).nonzero(as_tuple=True)[0]
            # print(f"class {cls}: {len(cls_idxs)}")
            if len(cls_idxs) < n_query + 1:
                raise ValueError(f"Not enough samples for class {cls.item()}: need > {n_query}, got {len(cls_idxs)}")
            query_idxs.append(cls_idxs[-n_query:])
            support_idxs.append(cls_idxs[:-n_query])

        support_idxs = torch.cat(support_idxs)
        query_idxs = torch.cat(query_idxs)

        support = input[support_idxs]
        query = input[query_idxs]

        prototypes = []
        for cls in classes:
            cls_support = support[(target[support_idxs] == cls)]
            prototypes.append(cls_support.mean(0))

            # Update stored prototypes for future use
            self.class_prototypes[cls.item()] = cls_support.mean(0).detach()
            self.seen_classes.add(cls.item())

        prototypes = torch.stack(prototypes)

        dists = euclidean_dist(query, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1)

        target_inds = torch.arange(n_classes, device=target.device).repeat_interleave(n_query)
        loss = -log_p_y[range(len(query_idxs)), target_inds].mean()

        pred = log_p_y.argmax(dim=1)
        acc = (pred == target_inds).float().mean()

        return loss, acc

    # def contrastive_loss(self, input, target, temperature=0.1):
    #     """
    #     Compute a supervised contrastive learning loss (InfoNCE).
    #
    #     Args:
    #         input (Tensor): Model embeddings of shape [N, D]
    #         target (Tensor): Ground truth labels of shape [N]
    #         temperature (float): Temperature scaling factor for softmax
    #
    #     Returns:
    #         Tuple[Tensor, Tensor]: (loss, accuracy)
    #     """
    #     classes = torch.unique(target)
    #     n_classes = len(classes)
    #
    #     # Normalize embeddings to use cosine similarity
    #     input = F.normalize(input, dim=1)
    #
    #     N, D = input.shape
    #     sim_matrix = torch.matmul(input, input.T)  # [N, N] pairwise similarities
    #     sim_matrix = sim_matrix / temperature
    #
    #     # Mask to ignore self-similarity
    #     self_mask = torch.eye(N, device=input.device).bool()
    #
    #     # Positive mask: 1 if same class, 0 otherwise
    #     target = target.contiguous()
    #     positive_mask = target.unsqueeze(0) == target.unsqueeze(1)  # [N, N]
    #     positive_mask = positive_mask & ~self_mask  # remove self-comparison
    #
    #     # For each sample, the denominator is similarity with all others
    #     # Only positives contribute to the numerator
    #     exp_sim = torch.exp(sim_matrix) * (~self_mask)  # exclude self
    #     log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    #
    #     # Compute loss: average over positives
    #     # Only consider valid positives (same class pairs)
    #     loss = -(log_prob * positive_mask).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-12)
    #     loss = loss.mean()
    #
    #     # Accuracy: treat nearest neighbor as prediction
    #     sim_matrix.masked_fill_(self_mask, -1e9)  # ignore self
    #     preds = sim_matrix.argmax(dim=1)
    #     acc = (target[preds] == target).float().mean()
    #
    #     for cls in classes:
    #         cls_support = input[(target == cls)]
    #
    #         # Update stored prototypes for future use
    #         self.class_prototypes[cls.item()] = cls_support.mean(0).detach()
    #         self.seen_classes.add(cls.item())
    #
    #     return loss, acc

    def contrastive_loss(self, input, target, temperature=0.1):
        """
        Compute a supervised contrastive learning loss (InfoNCE).

        Args:
            input (Tensor): Model embeddings of shape [N, D]
            target (Tensor): Ground truth labels of shape [N]
            temperature (float): Temperature scaling factor for softmax

        Returns:
            Tuple[Tensor, Tensor, Dict]: (loss, accuracy, class_losses)
        """
        classes = torch.unique(target)
        n_classes = len(classes)

        # Normalize embeddings to use cosine similarity
        input = F.normalize(input, dim=1)

        N, D = input.shape
        sim_matrix = torch.matmul(input, input.T) / temperature  # [N, N]

        # Mask to ignore self-similarity
        self_mask = torch.eye(N, device=input.device).bool()

        # Positive mask: 1 if same class, 0 otherwise
        target = target.contiguous()
        positive_mask = (target.unsqueeze(0) == target.unsqueeze(1)) & ~self_mask

        # Check if any sample has no positives
        num_positives = positive_mask.sum(dim=1)
        valid_samples = num_positives > 0

        # Numerical stability: log-sum-exp trick
        sim_matrix_masked = sim_matrix.clone()
        sim_matrix_masked[self_mask] = float('-inf')

        max_sim = sim_matrix_masked.max(dim=1, keepdim=True)[0]
        exp_sim = torch.exp(sim_matrix_masked - max_sim)
        log_sum_exp = max_sim + torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # Log probabilities
        log_prob = sim_matrix - log_sum_exp

        # Compute loss per sample
        loss_per_sample = -(log_prob * positive_mask).sum(dim=1) / (num_positives + 1e-12)

        # Store loss per class
        class_losses = {}
        for cls in classes:
            cls_mask = target == cls
            cls_samples = cls_mask & valid_samples

            if cls_samples.any():
                cls_loss = loss_per_sample[cls_samples].mean()
                class_losses[cls.item()] = cls_loss.item()
            else:
                class_losses[cls.item()] = 0.0

        # Overall loss
        loss = loss_per_sample[valid_samples].mean()

        # Accuracy: treat nearest neighbor as prediction
        sim_matrix_for_acc = sim_matrix.clone()
        sim_matrix_for_acc.masked_fill_(self_mask, -1e9)
        preds = sim_matrix_for_acc.argmax(dim=1)
        acc = (target[preds] == target).float().mean()

        # Update prototypes
        for cls in classes:
            cls_support = input[target == cls]
            self.class_prototypes[cls.item()] = cls_support.mean(0).detach()
            self.seen_classes.add(cls.item())

        return loss, acc, class_losses

    def generate_synthetic_prototypes(self):
        """Generate synthetic prototypical data for condensation."""
        device = next(self.net.parameters()).device

        print(f"generate synthetic prototypes/seen: {self.seen_classes}"
              f"/task: {self._task_iteration}/epoch:{self._epoch_iteration}")
        # FIXME At beginning of each task (task_idx + epoch = 0), init new learnable prototypes
        # FIXME By using self.image_syn, we can store the state + reset whenever it required
        print(f"past: {self._past_epoch} - epoch_model: {self.n_epoch_model}")
        if (self._past_epoch == self.n_epoch_model
                and self._epoch_iteration == 0):
            # Initialize synthetic images for seen classes
            self.image_syn = []
            self.label_syn = []

            for class_id in self.seen_classes:
                # Create synthetic image for this class
                syn_img = torch.randn(size=(self.spc, self.channel, self.im_size[0], self.im_size[1]),
                                      dtype=torch.float, requires_grad=True, device=device)
                self.image_syn.append(syn_img)

                # Create corresponding label
                syn_label = torch.tensor([class_id], dtype=torch.long, device=device)
                self.label_syn.append(syn_label)
                # print(f"class:{class_id} - syn_image:{syn_img}")
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

        # print(f"image_syn: {self.image_syn}")


    def optimize_proto_exemplar(self, image_syn, label_syn,
                                inputs = None, labels = None,
                                target_features=None):
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

        print(f"Optimize Exemplar")
        tmp_image_syn = []
        criterion_proto_align = nn.MSELoss()
        # Optimize each synthetic exemplar individually
        for i, (syn_img, syn_label) in enumerate(zip(image_syn, label_syn)):
            class_id = syn_label.item()

            # Skip if we haven't seen this class yet (no prototype to align to)
            if class_id not in self.class_prototypes:
                print(class_id)
                continue

            # Create optimizer for this specific synthetic image
            optimizer_img = Adam([syn_img], lr=self.args.lr_img)

            # target_proto = self.class_prototypes[class_id]
            # FIXME inputs -> rep. -> target_proto
            outputs, features = self.net(inputs, returnt = 'both')
            cls_support = features[(labels == class_id)]
            target_proto = cls_support.mean(0)

            # Optimize this synthetic exemplar over single step
            optimizer_img.zero_grad()
            total_loss = 0

            # Forward pass: get features from synthetic image
            # syn_img has shape (1, C, H, W) - single exemplar image
            syn_features = self.extract_features(syn_img)

            # syn_features has shape (1, feature_dim) - features from the single exemplar
            # The exemplar itself IS the prototype, so we squeeze the batch dimension
            syn_proto = syn_features.mean(0)  # Remove batch dimension: (feature_dim,)

            # Ensure target prototype has same shape
            if target_proto.dim() != syn_proto.dim():
                if target_proto.dim() > 1:
                    target_proto = target_proto.view(-1)
                if syn_proto.dim() > 1:
                    syn_proto = syn_proto.view(-1)

            # TODO Align synthetic prototype with stored / target prototype
            align_loss = criterion_proto_align(syn_proto, target_proto)
            # keep_loss = criterion_proto_align(syn_proto_1, buff_proto)
            total_loss = align_loss

            # Backward pass and optimization step
            total_loss.backward()
            optimizer_img.step()

            # Optional: Add some constraints to keep synthetic images realistic
            with torch.no_grad():
                syn_img.clamp_(-2.0, 2.025)  # Adjust range based on your data normalization

            tmp_image_syn.append(syn_img)

        print(f"average proto loss: {total_loss.item()}")

        # Update buffer with optimized synthetic exemplars
        self._add_synthetic_to_buffer(tmp_image_syn, label_syn)

    def _add_synthetic_to_buffer(self, image_syn, label_syn):
        """
        Add optimized synthetic exemplars to the buffer.

        Args:
            image_syn: List of optimized synthetic images
            label_syn: Corresponding labels
        """
        with torch.no_grad():
            for syn_img, syn_label in zip(image_syn, label_syn):
                # print(f"==== storing ====")
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
                Freeze Model + Train Data
        """

        if self.args.augment_flag == 1:
            inputs, labels = self.augmentation(inputs, labels)
        elif self.args.augment_flag == 2:
            inputs = torch.cat([inputs] + [not_aug_inputs], dim=0)
            labels = labels.repeat(1 + 1)
        elif self.args.augment_flag == 3:
            inputs, labels = not_aug_inputs, labels
        else:
            inputs, labels = inputs, labels

        # print(f"not aug: {not_aug_inputs.size()} | aug: {inputs.size()}")

        if epoch <= self.n_epoch_model -1:
            # Prototype + Network Optimizer
            self.opt.zero_grad()

            # Forward pass
            outputs, features = self.net(inputs, returnt = 'both')

            # Vanilla Loss (standard classification)
            vanilla_loss = self.loss(outputs, labels)

            if epoch % 100 == 0 and epoch != 0:
                self.proto_temp = self.proto_temp
            # ProtoNet Loss (prototypical network loss)
            if self.args.proto_method == 'contrastive':
                proto_loss, _, self.class_losses = self.contrastive_loss(features, labels, self.proto_temp)
            elif self.args.proto_method == 'prototypical':
                proto_loss, _ = self.prototypical_loss(features, labels, 20)
            elif self.args.proto_method == 'hyperbolic':
                proto_loss = self.hyperbolic_cl(features, labels)
                classes = torch.unique(labels)

                for cls in classes:
                    cls_support = features[(labels == cls)]

                    # Update stored prototypes for future use
                    self.class_prototypes[cls.item()] = cls_support.mean(0).detach()
                    self.seen_classes.add(cls.item())
            else:
                proto_loss = self.compute_prototype_loss(features, labels)

            # Combined loss
            loss = (1-self.args.alpha) * vanilla_loss + self.args.alpha * proto_loss

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
            print(f"Update Data epoch:{epoch} / epoch_model: {self.n_epoch_model}")
            self.generate_synthetic_prototypes()
            if self.image_syn:  # Only if we have synthetic prototypes
                self.optimize_proto_exemplar(image_syn = self.image_syn,
                                             label_syn = self.label_syn,
                                             inputs = inputs, labels= labels,
                                             )

            return 0


    def forward(self, x):
        """Forward pass through the network."""
        return self.net(x)


    def end_epoch(self, epoch: int, dataset: 'ContinualDataset') -> None:
        """
            - Prepares the model for the next epoch.
            - Visualize every epoch.
        """
        if self._current_task < 0:
            pass
        else:
            print(f"end epoch // epoch: {epoch} // e_model: {self.n_epoch_model}")
            if epoch % 25 != 0:
                # """ ========== Early Stopping & Best Model ========== """
                # # 1. Extract embeddings + labels from full dataset
                # all_features, all_labels = [], []
                # with torch.no_grad():
                #     # for k, test_loader in enumerate(dataset.test_loaders):
                #     for k, test_loader in enumerate([dataset.train_loader]):
                #         # for inputs, targets in test_loader:
                #         for inputs, targets, _ in test_loader:
                #             inputs, targets = inputs.to(self.net.device), targets.to(self.net.device)
                #             _, feats = self.net(inputs, returnt='both')
                #             all_features.append(feats.cpu())
                #             all_labels.append(targets.cpu())
                #
                # all_features = torch.cat(all_features, dim=0)
                # all_labels = torch.cat(all_labels, dim=0)
                #
                # # ===== ProtoNet Loss =====
                # if self.args.proto_method == 'contrastive':
                #     proto_loss, _, classes_losses = self.contrastive_loss(all_features, all_labels, self.args.proto_temp)
                #     print(f"Train Class Losses: {self.class_losses}")
                #     print(f"Evals Class Losses: {classes_losses}")
                # elif self.args.proto_method == 'prototypical':
                #     proto_loss, _ = self.prototypical_loss(all_features, all_labels, 20)
                # else:
                #     proto_loss = self.compute_prototype_loss(all_features, all_labels)
                #
                # # ===== Early Stopping Check =====
                # current_loss = proto_loss
                #
                # if current_loss < self.best_loss - self.args.min_delta:
                #     # Improvement detected
                #     self.best_loss = current_loss
                #     self.patience_counter = 0
                #
                #     # Save the current best model
                #     torch.save({
                #         'epoch': epoch,
                #         'model_state_dict': self.net.state_dict(),
                #         'optimizer_state_dict': self.opt.state_dict(),
                #         'loss': current_loss
                #     }, self.best_model_path)
                #
                #     print(f"[INFO] Best model saved at epoch {epoch} with loss {current_loss:.4f}")
                #
                #     # ===== Visualize the best case =====
                #     # 1. Extract embeddings + labels from full dataset
                #     all_features, all_labels = [], []
                #     with torch.no_grad():
                #         # for k, test_loader in enumerate(dataset.test_loaders):
                #         for k, test_loader in enumerate([dataset.train_loader]):
                #             # for inputs, targets in test_loader:
                #             for inputs, targets, _ in test_loader:
                #                 inputs, targets = inputs.to(self.net.device), targets.to(self.net.device)
                #                 _, feats = self.net(inputs, returnt='both')
                #                 all_features.append(feats.cpu())
                #                 all_labels.append(targets.cpu())
                #
                #     all_features = torch.cat(all_features, dim=0)
                #     all_labels = torch.cat(all_labels, dim=0)
                #
                #     syn_protos = []
                #     prototypes = []
                #     predictions = []
                #     num_classes = len(self.seen_classes)
                #     for class_id in self.seen_classes:
                #         class_mask = (all_labels == class_id)
                #         class_features = all_features[class_mask]
                #
                #         if len(class_features) > 0:
                #             proto = class_features.mean(dim=0)  # centroid of features
                #             print(
                #                 f"Difference train-test proto {class_id}: {torch.norm(self.class_prototypes[class_id].cpu() - proto)}")
                #             prototypes.append(proto)
                #             predictions_onehot = torch.nn.functional.one_hot(torch.tensor(class_id),
                #                                                              num_classes=num_classes).float()
                #             predictions.append(predictions_onehot)
                #         else:
                #             # if no samples of this class are present
                #             prototypes.append(torch.zeros(all_features.size(1)))
                #             predictions_onehot = torch.nn.functional.one_hot(torch.tensor(class_id),
                #                                                              num_classes=num_classes).float()
                #             predictions.append(predictions_onehot)
                #
                #         if class_id in self.buf_syn_img:
                #             if self.buf_syn_img[class_id] is not None:
                #                 _, feats = self.net(self.buf_syn_img[class_id], returnt='both')
                #                 syn_protos.append(feats.squeeze(0).cpu())
                #             else:
                #                 syn_protos.append(torch.zeros(self.spc, all_features.size(1)))
                #         else:
                #             syn_protos.append(torch.zeros(self.spc, all_features.size(1)))
                #
                #     prototypes = torch.stack(prototypes, dim=0)  # [num_classes, D]
                #     predictions = torch.stack(predictions, dim=0)
                #     syn_protos = torch.stack(syn_protos, dim=0)  # [num_classes, D]
                #
                #     # 3. Visualize with your method
                #     fig = self.visualizer.visualize_episode(
                #         embeddings=all_features,
                #         labels=all_labels,
                #         task=self._current_task,
                #         epoch=epoch,
                #         prototypes=prototypes,
                #         predictions=predictions,
                #         syn_proto=syn_protos,
                #         method="tsne",  # or "pca"
                #         title=f"Best TSNE: t{self._current_task}e{epoch}"
                #     )
                #     # fig = self.visualizer.visualize_episode(
                #     #     embeddings=all_features,
                #     #     labels=all_labels,
                #     #     task=self._current_task,
                #     #     epoch=epoch,
                #     #     prototypes=prototypes,
                #     #     predictions=predictions,
                #     #     syn_proto=syn_protos,
                #     #     method="humap",  # or "pca"
                #     #     title=f"Best HUMAP: t{self._current_task}e{epoch}"
                #     # )
                # else:
                #     # No improvement
                #     self.patience_counter += 1
                #     if self.patience_counter >= self.args.patience:
                #         print(f"[Early Stopping] No improvement for {self.args.patience} epochs.")
                #         stop_training = True
                pass
                """ ========== Early Stopping & Best Model ========== """
            else:
                # 1. Extract embeddings + labels from full dataset
                all_features, all_labels = [], []
                with torch.no_grad():
                    for k, test_loader in enumerate(dataset.test_loaders):
                    # for k, test_loader in enumerate([dataset.train_loader]):
                        # for inputs, targets in test_loader:
                        for inputs, targets, _ in test_loader:
                            inputs, targets = inputs.to(self.net.device), targets.to(self.net.device)
                            _, feats = self.net(inputs, returnt = 'both')
                            all_features.append(feats.cpu())
                            all_labels.append(targets.cpu())

                all_features = torch.cat(all_features, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                syn_protos = []
                prototypes = []
                predictions = []
                num_classes = len(self.seen_classes)
                for class_id in self.seen_classes:
                    class_mask = (all_labels == class_id)
                    class_features = all_features[class_mask]

                    if len(class_features) > 0:
                        proto = class_features.mean(dim=0)  # centroid of features
                        print(f"Difference train-test proto {class_id}: {torch.norm(self.class_prototypes[class_id].cpu() - proto)}")
                        prototypes.append(proto)
                        predictions_onehot = torch.nn.functional.one_hot(torch.tensor(class_id), num_classes=num_classes).float()
                        predictions.append(predictions_onehot)
                    else:
                        # if no samples of this class are present
                        prototypes.append(torch.zeros(all_features.size(1)))
                        predictions_onehot = torch.nn.functional.one_hot(torch.tensor(class_id), num_classes=num_classes).float()
                        predictions.append(predictions_onehot)

                    if class_id in self.buf_syn_img:
                        if self.buf_syn_img[class_id] is not None:
                            _, feats = self.net(self.buf_syn_img[class_id], returnt='both')
                            syn_protos.append(feats.squeeze(0).cpu())
                        else:
                            syn_protos.append(torch.zeros(self.spc, all_features.size(1)))
                    else:
                        syn_protos.append(torch.zeros(self.spc, all_features.size(1)))

                prototypes = torch.stack(prototypes, dim=0)  # [num_classes, D]
                predictions = torch.stack(predictions, dim=0)
                syn_protos = torch.stack(syn_protos, dim=0)  # [num_classes, D]

                # 3. Visualize with your method
                fig = self.visualizer.visualize_episode(
                    embeddings=all_features,
                    labels=all_labels,
                    task=self._current_task,
                    epoch=epoch,
                    prototypes=prototypes,
                    predictions=predictions,
                    syn_proto = syn_protos,
                    method="tsne",  # or "pca" or "tsne"
                    title=f"t-SNE of t{self._current_task}e{epoch}"
                )
                # fig = self.visualizer.visualize_episode(
                #     embeddings=all_features,
                #     labels=all_labels,
                #     task=self._current_task,
                #     epoch=epoch,
                #     prototypes=prototypes,
                #     predictions=predictions,
                #     syn_proto=syn_protos,
                #     method="humap",  # or "pca" or "tsne"
                #     title=f"HUMAP of t{self._current_task}e{epoch}"
                # )
                # plt.savefig(f"task{self._current_task}-epoch{epoch}.png")  # saves to file
                # plt.close()