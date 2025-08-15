# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


class ProtoDCC(ContinualModel):
    """Continual learning via Prototype Set Condensation."""
    NAME = 'protodcc'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        #TODO Prototype + Network Optimizer
        self.opt.zero_grad()


        #TODO Condensation Optimizer
        image_syn = [torch.randn(size=(1, channel, im_size[0], im_size[1]), dtype=torch.float,
                                requires_grad=True, device=args.device) for i in range(self.num_classes)]
        label_syn = [torch.tensor(i, dtype=torch.long, requires_grad=False,
                                  device=args.device).view(-1) for i in range(num_classes)] # [0,0,0, 1,1,1, ..., 9,9,9]

        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion_proto_align = nn.MSELoss()

        #TODO Vanilla Loss + ProtoNet Loss
        outputs = self.net(inputs)
        #TODO Vanilla Loss
        vanilla_loss = self.loss(outputs, labels)
        #TODO ProtoNet Loss
        proto_loss = self.loss(outputs, labels)

        loss = vanilla_loss + self.args.alpha

        if not self.buffer.is_empty():
            #TODO These are Prototypical Exemplar (get from Buffer)
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)

            #TODO Learned Prototype vs. Stored Prototype
            loss_pa = self.args.beta * criterion_proto_align()
            loss += loss_pa

            #TODO Align Prototypical Network


        loss.backward()
        self.opt.step()

        #TODO Add Prototypical Exemplar here
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return loss.item()

    def optimize_proto_exemplar(self, image_syn, label_syn, target_features):
        """ Optimize synthetic prototypes to match target features.


        """
        if not image_syn:
            return

        # Optimizer for synthetic images
        for i, (syn_img, syn_label) in enumerate(zip(image_syn, label_syn)):
            optimizer_img = SGD([{'params': syn_img, 'lr': self.args.lr_img}],
                                momentum=0.5)

            criterion_proto_align = nn.MSELoss()

            for step in range(self.proto_steps):
                optimizer_img.zero_grad()

                # Get features from synthetic image
                syn_proto = self.extract_features(syn_img)

                # Get target prototype for this class
                class_id = syn_label.item()
                if class_id in self.class_prototypes:
                    target_proto = self.class_prototypes[class_id]

                    # Align synthetic features with target prototype
                    align_loss = criterion_proto_align(syn_proto, target_proto)

                    if align_loss > 0:
                        align_loss.backward()
                        optimizer_img.step()

        # Add prototypical exemplars to buffer