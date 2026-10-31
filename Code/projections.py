# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#               2021 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#               2023 Bing Han (hanbing97@sjtu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_projection(conf):
    if conf['project_type'] == 'add_margin':
        projection = AddMarginProduct(conf['embed_dim'],
                                      conf['num_class'],
                                      scale=conf['scale'],
                                      margin=0.0)
    elif conf['project_type'] == 'arc_margin':
        projection = ArcMarginProduct(conf['embed_dim'],
                                      conf['num_class'],
                                      scale=conf['scale'],
                                      margin=0.0,
                                      easy_margin=conf['easy_margin'])
    elif conf['project_type'] == 'elastic_arc_margin':

        projection = ElasticArcMarginProduct(
            conf['embed_dim'],
            conf['num_class'],

            scale=conf.get(
                'scale',
                32.0
            ),

            # IMPORTANT:
            # still controlled by MarginScheduler
            margin=0.0,

            easy_margin=conf.get(
                'easy_margin',
                False
            ),

            margin_std=conf.get(
                'margin_std',
                0.015
            ),

            reference_margin=conf.get(
                'reference_margin',
                0.20
            ),

            max_margin=conf.get(
                'max_margin',
                0.30
            )
        )
    
    elif conf['project_type'] == 'curricular_arc_margin':

        projection = CurricularArcMarginProduct(
            conf['embed_dim'],
            conf['num_class'],

            scale=conf.get(
                'scale',
                32.0
            ),

            # still controlled by MarginScheduler
            margin=0.0,

            easy_margin=conf.get(
                'easy_margin',
                False
            ),

            t_alpha=conf.get(
                't_alpha',
                0.01
            )
        )
    elif conf['project_type'] == 'confidence_scale_arc_margin':

        projection = ConfidenceScaleArcMarginProduct(
            conf['embed_dim'],
            conf['num_class'],

            scale=conf.get(
                'scale',
                32.0
            ),

            margin=0.0,

            easy_margin=conf.get(
                'easy_margin',
                False
            ),

            beta=conf.get(
                'beta',
                0.25
            ),

            confidence_tau=conf.get(
                'confidence_tau',
                0.10
            ),

            reference_margin=conf.get(
                'reference_margin',
                0.20
            ),

            min_scale=conf.get(
                'min_scale',
                20.0
            )
        )
    elif conf['project_type']=="subcenter_arc_margin":

        projection = SubCenterArcMarginProduct(
            conf['embed_dim'],
            conf['num_class'],

            scale=conf.get(
                'scale',
                32.0
            ),

            margin=0.0,

            K=conf.get(
                'K',
                3
            ),

            easy_margin=conf.get(
                'easy_margin',
                False
            )
        )
    elif conf['project_type'] == 'real_am_margin':

        projection = RealAMSoftmax(
            conf['embed_dim'],
            conf['num_class'],
            scale=conf.get('scale', 32.0),

            # IMPORTANT:
            # MarginScheduler will initialize/update it.
            margin=0.0
        )
    elif conf['project_type'] == 'arc_margin_intertopk_subcenter':
        projection = ArcMarginProduct_intertopk_subcenter(
            conf['embed_dim'],
            conf['num_class'],
            scale=conf['scale'],
            margin=0.0,
            easy_margin=conf['easy_margin'],
            K=conf.get('K', 3),
            mp=conf.get('mp', 0.06),
            k_top=conf.get('k_top', 5),
            do_lm=conf.get('do_lm', False))
    elif conf['project_type'] == 'sphere':
        projection = SphereProduct(conf['embed_dim'],
                                   conf['num_class'],
                                   margin=4)
    elif conf['project_type'] == 'sphereface2':
        projection = SphereFace2(conf['embed_dim'],
                                 conf['num_class'],
                                 scale=conf['scale'],
                                 margin=0.0,
                                 t=conf.get('t', 3),
                                 lanbuda=conf.get('lanbuda', 0.7),
                                 margin_type=conf.get('margin_type', 'C'))
    elif conf['project_type'] == 'confidence_calibrated_arc_margin':

        projection = ConfidenceCalibratedArcMarginProduct(
        conf['embed_dim'],
        conf['num_class'],
        scale=conf.get('scale',32.0),
        margin=0.0,
        easy_margin=conf.get('easy_margin',False),
        confidence_alpha=conf.get('confidence_alpha',0.1),
        confidence_tau=conf.get('confidence_tau',0.1)
    )
    else:
        projection = Linear(conf['embed_dim'], conf['num_class'])

    return projection

class ConfidenceScaleArcMarginProduct(nn.Module):
    """
    Confidence-Modulated Scale AAM-Softmax.

    Keep the standard AAM angular margin unchanged.

    Only modulate the sample-wise scale according to
    target-vs-hardest-negative confidence.

    Easy / reliable samples:
        scale_i -> base scale

    Hard / ambiguous samples:
        scale_i decreases

    The modulation is gradually activated together with
    WeSpeaker MarginScheduler.
    """

    def __init__(
        self,
        in_features,
        out_features,
        scale=32.0,
        margin=0.2,
        easy_margin=False,

        beta=0.25,
        confidence_tau=0.10,
        reference_margin=0.20,

        min_scale=20.0,
        eps=1e-7,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        self.beta = beta
        self.confidence_tau = confidence_tau
        self.reference_margin = reference_margin
        self.min_scale = min_scale

        self.eps = eps

        self.weight = nn.Parameter(
            torch.FloatTensor(
                out_features,
                in_features
            )
        )

        nn.init.xavier_uniform_(self.weight)

        self.m = margin

        self._update_margin_parameters()


    def _update_margin_parameters(self):

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)

        self.th = math.cos(
            math.pi - self.margin
        )

        self.mmm = (
            1.0
            + math.cos(
                math.pi - self.margin
            )
        )


    def update(self, margin=0.2):

        self.margin = margin
        self.m = margin

        self._update_margin_parameters()


    def forward(self, input, label):

        # =====================================================
        # 1. Standard normalized cosine
        # =====================================================

        cosine = F.linear(
            F.normalize(input, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1)
        )

        cosine = cosine.clamp(
            -1.0 + self.eps,
            1.0 - self.eps
        )


        # =====================================================
        # 2. Target cosine
        # =====================================================

        target_cosine = cosine.gather(
            1,
            label.view(-1, 1).long()
        )


        # =====================================================
        # 3. Standard AAM positive margin
        #
        # EXACTLY preserve ArcMargin
        # =====================================================

        target_sine = torch.sqrt(
            torch.clamp(
                1.0 - target_cosine.pow(2),
                min=self.eps
            )
        )

        phi = (
            target_cosine * self.cos_m
            - target_sine * self.sin_m
        )


        if self.easy_margin:

            final_target = torch.where(
                target_cosine > 0,
                phi,
                target_cosine
            )

        else:

            final_target = torch.where(
                target_cosine > self.th,
                phi,
                target_cosine - self.mmm
            )


        # =====================================================
        # 4. Estimate sample difficulty
        #
        # IMPORTANT:
        # use RAW cosine before AAM margin
        # =====================================================

        with torch.no_grad():

            target_raw = target_cosine.squeeze(1)


            target_mask = torch.zeros_like(
                cosine,
                dtype=torch.bool
            )

            target_mask.scatter_(
                1,
                label.view(-1, 1).long(),
                True
            )


            negative_cosine = cosine.masked_fill(
                target_mask,
                -2.0
            )


            hardest_negative = (
                negative_cosine
                .max(dim=1)
                .values
            )


            # ---------------------------------------------
            # hardness
            #
            # easy -> 0
            # boundary -> 0.5
            # incorrect -> 1
            # ---------------------------------------------

            hardness = torch.sigmoid(
                (
                    hardest_negative
                    - target_raw
                )
                / self.confidence_tau
            )


            # =================================================
            # 5. Gradually activate scale modulation
            # =================================================

            if self.reference_margin > 0:

                progress = min(
                    max(
                        self.margin
                        / self.reference_margin,
                        0.0
                    ),
                    1.0
                )

            else:

                progress = 1.0


            current_beta = (
                self.beta
                * progress
            )


            # =================================================
            # 6. Sample-dependent scale
            #
            # easy:
            #     ~32
            #
            # difficult:
            #     lower scale
            # =================================================

            sample_scale = (
                self.scale
                * (
                    1.0
                    - current_beta
                    * hardness
                )
            )


            sample_scale = torch.clamp(
                sample_scale,
                min=self.min_scale,
                max=self.scale
            )


        sample_scale = sample_scale.unsqueeze(1)


        # =====================================================
        # 7. Standard AAM logits
        # =====================================================

        output = cosine.clone()

        output.scatter_(
            1,
            label.view(-1, 1).long(),
            final_target
        )


        # =====================================================
        # 8. Apply sample-wise scale
        # =====================================================

        output = (
            output
            * sample_scale
        )

        return output


    def extra_repr(self):

        return (
            'in_features={}, '
            'out_features={}, '
            'scale={}, '
            'margin={}, '
            'easy_margin={}, '
            'beta={}, '
            'confidence_tau={}, '
            'min_scale={}'
        ).format(
            self.in_features,
            self.out_features,
            self.scale,
            self.margin,
            self.easy_margin,
            self.beta,
            self.confidence_tau,
            self.min_scale
        )
    
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class CurricularArcMarginProduct(nn.Module):
    """
    Curricular AAM-Softmax.

    Standard AAM positive logit:
        cos(theta_y + m)

    Hard-negative modulation:
        cos(theta_j) -> cos(theta_j) * (t + cos(theta_j))

    where t is an EMA of the target cosine similarity.

    Early training:
        t is small -> hard negatives are relatively suppressed.

    Later training:
        t increases -> hard negatives receive stronger emphasis.
    """

    def __init__(
        self,
        in_features,
        out_features,
        scale=32.0,
        margin=0.2,
        easy_margin=False,
        t_alpha=0.01,
        eps=1e-7,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # EMA update rate
        self.t_alpha = t_alpha

        self.eps = eps

        self.weight = nn.Parameter(
            torch.FloatTensor(
                out_features,
                in_features
            )
        )

        nn.init.xavier_uniform_(self.weight)

        # global curriculum state
        self.register_buffer(
            "t",
            torch.zeros(1)
        )

        # for WeSpeaker MarginScheduler
        self.m = margin

        self._update_margin_parameters()


    def _update_margin_parameters(self):

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)

        self.th = math.cos(
            math.pi - self.margin
        )

        self.mmm = (
            1.0
            + math.cos(
                math.pi - self.margin
            )
        )


    def update(self, margin=0.2):
        """
        Compatible with WeSpeaker MarginScheduler.
        """

        self.margin = margin
        self.m = margin

        self._update_margin_parameters()


    def forward(self, input, label):

        # =====================================================
        # 1. Normalized cosine classifier
        # =====================================================

        cosine = F.linear(
            F.normalize(input, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1)
        )

        cosine = torch.clamp(
            cosine,
            -1.0 + self.eps,
            1.0 - self.eps
        )


        # =====================================================
        # 2. Ground-truth cosine
        # =====================================================

        target_cosine = cosine.gather(
            1,
            label.view(-1, 1).long()
        )


        # =====================================================
        # 3. Standard AAM positive logit
        #
        # cos(theta_y + m)
        # =====================================================

        target_sine = torch.sqrt(
            torch.clamp(
                1.0 - target_cosine.pow(2),
                min=self.eps
            )
        )

        phi = (
            target_cosine * self.cos_m
            - target_sine * self.sin_m
        )


        if self.easy_margin:

            final_target = torch.where(
                target_cosine > 0,
                phi,
                target_cosine
            )

        else:

            final_target = torch.where(
                target_cosine > self.th,
                phi,
                target_cosine - self.mmm
            )


        # =====================================================
        # 4. Update curriculum parameter t
        # =====================================================

        with torch.no_grad():

            batch_target_mean = target_cosine.mean()

            # synchronize between GPUs
            if (
                dist.is_available()
                and dist.is_initialized()
            ):

                batch_target_mean_sync = (
                    batch_target_mean.clone()
                )

                dist.all_reduce(
                    batch_target_mean_sync,
                    op=dist.ReduceOp.SUM
                )

                batch_target_mean_sync /= (
                    dist.get_world_size()
                )

                batch_target_mean = (
                    batch_target_mean_sync
                )

            # EMA
            self.t.mul_(
                1.0 - self.t_alpha
            ).add_(
                self.t_alpha
                * batch_target_mean
            )


        # =====================================================
        # 5. Find hard negative classes
        #
        # hard negative means:
        #
        # cos(theta_j) > cos(theta_y + m)
        #
        # i.e. the negative class violates the AAM boundary.
        # =====================================================

        target_mask = torch.zeros_like(
            cosine,
            dtype=torch.bool
        )

        target_mask.scatter_(
            1,
            label.view(-1, 1).long(),
            True
        )


        hard_mask = (
            cosine > final_target
        ) & (~target_mask)


        # =====================================================
        # 6. Curriculum modulation of hard negatives
        #
        # Early:
        #     t small
        #
        # Later:
        #     t grows
        #
        # More confusing negative:
        #     larger cosine
        #     -> larger modulation
        # =====================================================

        hard_example = cosine[hard_mask]

        cosine = cosine.clone()

        cosine[hard_mask] = (
            hard_example
            * (
                self.t
                + hard_example
            )
        )


        # =====================================================
        # 7. Replace target class with standard AAM logit
        # =====================================================

        cosine.scatter_(
            1,
            label.view(-1, 1).long(),
            final_target
        )


        # =====================================================
        # 8. Scale
        # =====================================================

        output = cosine * self.scale

        return output


    def extra_repr(self):

        return (
            'in_features={}, '
            'out_features={}, '
            'scale={}, '
            'margin={}, '
            'easy_margin={}, '
            't_alpha={}, '
            't={:.4f}'
        ).format(
            self.in_features,
            self.out_features,
            self.scale,
            self.margin,
            self.easy_margin,
            self.t_alpha,
            self.t.item()
        )

class SphereFace2(nn.Module):
    r"""Implement of sphereface2 for speaker verification:
        Reference:
            [1] Exploring Binary Classification Loss for Speaker Verification
            https://ieeexplore.ieee.org/abstract/document/10094954
            [2] Sphereface2: Binary classification is all you need
            for deep face recognition
            https://arxiv.org/pdf/2108.01513
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            scale: norm of input feature
            margin: margin
            lanbuda: weight of positive and negative pairs
            t: parameter for adjust score distribution
            margin_type: A:cos(theta+margin) or C:cos(theta)-margin
        Recommend margin:
            training: 0.2 for C and 0.15 for A
            LMF: 0.3 for C and 0.25 for A
        """

    def __init__(self,
                 in_features,
                 out_features,
                 scale=32.0,
                 margin=0.2,
                 lanbuda=0.7,
                 t=3,
                 margin_type='C'):
        super(SphereFace2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(out_features,
                                                     in_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(1, 1))
        self.t = t
        self.lanbuda = lanbuda
        self.margin_type = margin_type

        ########
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)
        ########

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)

    def fun_g(self, z, t: int):
        gz = 2 * torch.pow((z + 1) / 2, t) - 1
        return gz

    def forward(self, input, label):
        # compute similarity
        cos = F.linear(F.normalize(input), F.normalize(self.weight))

        if self.margin_type == 'A':  # arcface type
            sin = torch.sqrt(1.0 - torch.pow(cos, 2))
            cos_m_theta_p = self.scale * self.fun_g(
                torch.where(cos > self.th, cos * self.cos_m - sin * self.sin_m,
                            cos - self.mmm), self.t) + self.bias[0][0]
            cos_m_theta_n = self.scale * self.fun_g(
                cos * self.cos_m + sin * self.sin_m, self.t) + self.bias[0][0]
            cos_p_theta = self.lanbuda * torch.log(
                1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (
                1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))
        else:  # cosface type
            cos_m_theta_p = self.scale * (self.fun_g(cos, self.t) -
                                          self.margin) + self.bias[0][0]
            cos_m_theta_n = self.scale * (self.fun_g(cos, self.t) +
                                          self.margin) + self.bias[0][0]
            cos_p_theta = self.lanbuda * torch.log(
                1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (
                1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))

        target_mask = input.new_zeros(cos.size())
        target_mask.scatter_(1, label.view(-1, 1).long(), 1.0)
        nontarget_mask = 1 - target_mask
        cos1 = (cos - self.margin) * target_mask + cos * nontarget_mask
        output = self.scale * cos1  # for computing the accuracy
        loss = (target_mask * cos_p_theta +
                nontarget_mask * cos_n_theta).sum(1).mean()
        return output, loss

    def extra_repr(self):
        return '''in_features={}, out_features={}, scale={}, lanbuda={},
                  margin={}, t={}, margin_type={}'''.format(
            self.in_features, self.out_features, self.scale, self.lanbuda,
            self.margin, self.t, self.margin_type)


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            scale: norm of input feature
            margin: margin
            cos(theta + margin)
        """

    def __init__(self,
                 in_features,
                 out_features,
                 scale=32.0,
                 margin=0.2,
                 easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features,
                                                     in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(
            math.pi - margin)  # this can make the output more continuous
        ########
        self.m = self.margin
        ########

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)
        # self.weight = self.weight
        # self.scale = self.scale

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            ########
            # phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)
            ########

        one_hot = input.new_zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output

    def extra_repr(self):
        return '''in_features={}, out_features={}, scale={},
                  margin={}, easy_margin={}'''.format(self.in_features,
                                                      self.out_features,
                                                      self.scale, self.margin,
                                                      self.easy_margin)



class RealAMSoftmax(nn.Module):
    r"""
    Real Additive Margin Softmax for WeSpeaker.

    This implementation returns:
        1. normal cosine classification logits for accuracy logging
        2. RAM-Softmax loss for optimization

    RAM loss:
        L_i = log(
            1 + sum_{j != y_i}
            exp(
                max(
                    0,
                    s * (cos(theta_j) - cos(theta_y) + m)
                )
            )
        )

    Args:
        in_features: embedding dimension
        out_features: number of classes
        scale: scale factor s
        margin: real margin m
    """

    def __init__(
        self,
        in_features,
        out_features,
        scale=32.0,
        margin=0.0,
    ):
        super(RealAMSoftmax, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features)
        )

        nn.init.xavier_uniform_(self.weight)

        # WeSpeaker compatibility
        self.m = margin


    def update(self, margin=0.2):
        """
        Called by MarginScheduler.
        """
        self.margin = margin
        self.m = margin


    def forward(self, input, label):

        # =====================================================
        # 1. Normalized cosine classifier
        # =====================================================

        cosine = F.linear(
            F.normalize(input, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1)
        )

        cosine = torch.clamp(
            cosine,
            min=-1.0 + 1e-7,
            max=1.0 - 1e-7
        )

        # =====================================================
        # 2. Normal classification logits
        #
        # IMPORTANT:
        # These logits are ONLY used by WeSpeaker to compute
        # meaningful training accuracy.
        # They are NOT used to compute the RAM loss.
        # =====================================================

        output = cosine * self.scale

        # =====================================================
        # 3. Target cosine
        # =====================================================

        target_cosine = cosine.gather(
            1,
            label.view(-1, 1).long()
        )

        # shape:
        # [B, 1]

        # =====================================================
        # 4. Real margin violation
        #
        # s * (cos_j - cos_y + m)
        # =====================================================

        violation = self.scale * (
            cosine
            - target_cosine
            + self.margin
        )

        # =====================================================
        # 5. Remove target class
        # =====================================================

        target_mask = torch.zeros_like(
            cosine,
            dtype=torch.bool
        )

        target_mask.scatter_(
            1,
            label.view(-1, 1).long(),
            True
        )

        # =====================================================
        # 6. Real max-margin operation
        #
        # Easy negative:
        #
        # cos_y - cos_j >= m
        #
        # -> violation <= 0
        # -> gradient blocked
        #
        # Hard negative:
        #
        # cos_y - cos_j < m
        #
        # -> violation > 0
        # -> optimized
        # =====================================================

        violation = F.relu(violation)

        # target class does not belong to negative set
        violation = violation.masked_fill(
            target_mask,
            float('-inf')
        )

        # =====================================================
        # 7. Stable RAM loss
        #
        # log(
        #     1 +
        #     sum exp(violation_j)
        # )
        #
        # The leading "1" is represented by exp(0).
        # =====================================================

        zero = torch.zeros(
            violation.size(0),
            1,
            device=violation.device,
            dtype=violation.dtype
        )

        ram_terms = torch.cat(
            [zero, violation],
            dim=1
        )

        loss = torch.logsumexp(
            ram_terms,
            dim=1
        ).mean()

        # =====================================================
        # WeSpeaker executor already supports this:
        #
        # if isinstance(outputs, tuple):
        #     outputs, loss = outputs
        # =====================================================

        return output, loss


    def extra_repr(self):

        return (
            'in_features={}, out_features={}, '
            'scale={}, margin={}'
        ).format(
            self.in_features,
            self.out_features,
            self.scale,
            self.margin
        )

class ArcMarginProduct_intertopk_subcenter(nn.Module):
    r"""Implement of large margin arc distance with intertopk and subcenter:
        Reference:
            MULTI-QUERY MULTI-HEAD ATTENTION POOLING AND INTER-TOPK PENALTY
            FOR SPEAKER VERIFICATION.
            https://arxiv.org/pdf/2110.05042.pdf
            Sub-center ArcFace: Boosting Face Recognition by
            Large-Scale Noisy Web Faces.
            https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            scale: norm of input feature
            margin: margin
            cos(theta + margin)
            K: number of sub-centers
            k_top: number of hard samples
            mp: margin penalty of hard samples
            do_lm: whether do large margin finetune
        """

    def __init__(self,
                 in_features,
                 out_features,
                 scale=32.0,
                 margin=0.2,
                 easy_margin=False,
                 K=3,
                 mp=0.06,
                 k_top=5,
                 do_lm=False):
        super(ArcMarginProduct_intertopk_subcenter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.do_lm = do_lm

        # intertopk + subcenter
        self.K = K
        if do_lm:  # if do LMF, remove hard sample penalty
            self.mp = 0.0
            self.k_top = 0
        else:
            self.mp = mp
            self.k_top = k_top

        # initial classifier
        self.weight = nn.Parameter(
            torch.FloatTensor(self.K * out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(
            math.pi - margin)  # this can make the output more continuous
        ########
        self.m = self.margin
        ########
        self.cos_mp = math.cos(0.0)
        self.sin_mp = math.sin(0.0)

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)

        # hard sample margin is increasing as margin
        if margin > 0.001:
            mp = self.mp * (margin / 0.2)
        else:
            mp = 0.0
        self.cos_mp = math.cos(mp)
        self.sin_mp = math.sin(mp)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input),
                          F.normalize(self.weight))  # (batch, out_dim * k)
        cosine = torch.reshape(
            cosine, (-1, self.out_features, self.K))  # (batch, out_dim, k)
        cosine, _ = torch.max(cosine, 2)  # (batch, out_dim)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi_mp = cosine * self.cos_mp + sine * self.sin_mp

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            ########
            # phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)
            ########

        one_hot = input.new_zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.k_top > 0:
            # topk (j != y_i)
            _, top_k_index = torch.topk(cosine - 2 * one_hot,
                                        self.k_top)  # exclude j = y_i
            top_k_one_hot = input.new_zeros(cosine.size()).scatter_(
                1, top_k_index, 1)

            # sum
            output = (one_hot * phi) + (top_k_one_hot * phi_mp) + (
                (1.0 - one_hot - top_k_one_hot) * cosine)
        else:
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, scale={}, margin={}, easy_margin={},' \
            'K={}, mp={}, k_top={}, do_lm={}'.format(
                self.in_features, self.out_features, self.scale, self.margin,
                self.easy_margin, self.K, self.mp, self.k_top, self.do_lm)


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale: norm of input feature
        margin: margin
        cos(theta) - margin
    """

    def __init__(self, in_features, out_features, scale=32.0, margin=0.20):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features,
                                                     in_features))
        nn.init.xavier_uniform_(self.weight)

    def update(self, margin):
        self.margin = margin

    def forward(self, input, label):
        # ---------------- cos(theta) & phi(theta) ---------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.margin
        # ---------------- convert label to one-hot ---------------
        one_hot = input.new_zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', scale=' + str(self.scale) \
            + ', margin=' + str(self.margin) + ')'


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        margin: margin
        cos(margin * theta)
    """

    def __init__(self, in_features, out_features, margin=2):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = nn.Parameter(torch.FloatTensor(out_features,
                                                     in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x**0, lambda x: x**1, lambda x: 2 * x**2 - 1,
            lambda x: 4 * x**3 - 3 * x, lambda x: 8 * x**4 - 8 * x**2 + 1,
            lambda x: 16 * x**5 - 20 * x**3 + 5 * x
        ]
        assert self.margin < 6

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(
            self.LambdaMin,
            self.base * (1 + self.gamma * self.iter)**(-1 * self.power))

        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.margin](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.margin * theta / 3.14159265).floor()
        phi_theta = ((-1.0)**k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)
        one_hot = input.new_zeros(cos_theta.size())
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * (phi_theta - cos_theta) /
                  (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', margin=' + str(self.margin) + ')'


import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class SubCenterArcMarginProduct(nn.Module):
    """
    Sub-center AAM-Softmax

    Each class owns K prototypes.

    For target class:
        choose the closest sub-center

    For negative classes:
        use maximum similarity among sub-centers

    """

    def __init__(
            self,
            in_features,
            out_features,
            scale=32.0,
            margin=0.2,
            K=3,
            easy_margin=False):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.scale = scale
        self.margin = margin
        self.K = K
        self.easy_margin = easy_margin


        # K centers for every class
        self.weight = nn.Parameter(
            torch.FloatTensor(
                out_features*K,
                in_features
            )
        )


        nn.init.xavier_uniform_(
            self.weight
        )


        self.update_margin(margin)



    def update_margin(self, margin):

        self.margin = margin

        self.cos_m = math.cos(
            margin
        )

        self.sin_m = math.sin(
            margin
        )

        self.th = math.cos(
            math.pi-margin
        )

        self.mm = math.sin(
            math.pi-margin
        )*margin



    # compatible with WeSpeaker MarginScheduler
    def update(self, margin):

        self.update_margin(
            margin
        )



    def forward(self, input, label):


        # ================================
        # normalize
        # ================================

        x = F.normalize(
            input,
            dim=1
        )


        W = F.normalize(
            self.weight,
            dim=1
        )


        B = x.size(0)



        # [B, class*K]

        cosine = F.linear(
            x,
            W
        )


        # reshape:

        cosine = cosine.view(
            B,
            self.out_features,
            self.K
        )


        # ================================
        # select closest sub-center
        # ================================

        cosine, _ = torch.max(
            cosine,
            dim=2
        )


        # cosine:
        # [B,num_class]



        # ================================
        # AAM margin
        # ================================

        sine = torch.sqrt(
            1.0 -
            torch.pow(
                cosine,
                2
            )
            +
            1e-7
        )


        phi = (
            cosine*self.cos_m
            -
            sine*self.sin_m
        )


        if self.easy_margin:

            phi = torch.where(
                cosine>0,
                phi,
                cosine
            )

        else:

            phi = torch.where(
                cosine-self.th>0,
                phi,
                cosine-self.mm
            )



        one_hot = torch.zeros_like(
            cosine
        )


        one_hot.scatter_(
            1,
            label.view(-1,1),
            1
        )


        output = (
            one_hot*phi
            +
            (1-one_hot)*cosine
        )


        output *= self.scale


        return output

class ElasticArcMarginProduct(nn.Module):
    """
    Mean-Preserving Elastic AAM-Softmax.

    Instead of using a fixed angular margin m for every sample,
    sample-wise margins are drawn around the current base margin:

        m_i = m_base + sigma_t * epsilon_i

    where epsilon ~ N(0, 1).

    The random noise is centered within each batch so that:

        mean(m_i) ~= m_base

    Therefore the average AAM decision boundary is preserved.

    MarginScheduler still controls m_base.
    """

    def __init__(
        self,
        in_features,
        out_features,
        scale=32.0,
        margin=0.2,
        easy_margin=False,

        margin_std=0.015,
        reference_margin=0.20,
        max_margin=0.30,

        eps=1e-7,
    ):
        super(ElasticArcMarginProduct, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Maximum std when base margin reaches reference_margin
        self.margin_std = margin_std

        # Usually equal to final_margin
        self.reference_margin = reference_margin

        # Safety bound
        self.max_margin = max_margin

        self.eps = eps

        self.weight = nn.Parameter(
            torch.FloatTensor(
                out_features,
                in_features
            )
        )

        nn.init.xavier_uniform_(self.weight)

        # compatibility with WeSpeaker MarginScheduler
        self.m = margin


    def update(self, margin=0.2):
        """
        Called by WeSpeaker MarginScheduler.

        margin is still the BASE margin.
        """

        self.margin = margin
        self.m = margin


    def forward(self, input, label):

        # =====================================================
        # 1. Standard cosine classifier
        # =====================================================

        cosine = F.linear(
            F.normalize(input, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1)
        )

        cosine = torch.clamp(
            cosine,
            -1.0 + self.eps,
            1.0 - self.eps
        )

        batch_size = cosine.size(0)


        # =====================================================
        # 2. Generate sample-wise elastic margin
        # =====================================================

        if self.training and self.margin > 0.0:

            # ---------------------------------------------
            # Follow the existing WeSpeaker margin schedule
            #
            # m_base = 0
            # -> sigma = 0
            #
            # m_base = 0.1
            # -> sigma ~= 0.0075
            #
            # m_base = 0.2
            # -> sigma = 0.015
            # ---------------------------------------------

            progress = min(
                max(
                    self.margin / self.reference_margin,
                    0.0
                ),
                1.0
            )

            current_std = (
                self.margin_std
                * progress
            )


            # Gaussian random noise
            noise = torch.randn(
                batch_size,
                1,
                device=cosine.device,
                dtype=cosine.dtype
            )


            # ---------------------------------------------
            # IMPORTANT:
            # force zero mean inside each batch
            #
            # This keeps the batch-average margin close
            # to the original AAM margin.
            # ---------------------------------------------

            noise = noise - noise.mean()


            sample_margin = (
                self.margin
                + current_std * noise
            )


            # Prevent extreme or invalid margins
            sample_margin = torch.clamp(
                sample_margin,
                min=0.0,
                max=self.max_margin
            )

        else:

            # deterministic mean margin
            sample_margin = torch.full(
                (batch_size, 1),
                self.margin,
                device=cosine.device,
                dtype=cosine.dtype
            )


        # =====================================================
        # 3. Sample-wise ArcFace parameters
        # =====================================================

        cos_m = torch.cos(
            sample_margin
        )

        sin_m = torch.sin(
            sample_margin
        )


        sine = torch.sqrt(
            torch.clamp(
                1.0 - cosine.pow(2),
                min=self.eps
            )
        )


        # =====================================================
        # 4. Elastic AAM
        #
        # cos(theta + m_i)
        # =====================================================

        phi = (
            cosine * cos_m
            -
            sine * sin_m
        )


        # =====================================================
        # 5. Same monotonicity handling as WeSpeaker AAM
        # =====================================================

        if self.easy_margin:

            phi = torch.where(
                cosine > 0,
                phi,
                cosine
            )

        else:

            # sample-wise thresholds
            th = torch.cos(
                math.pi - sample_margin
            )

            mmm = (
                1.0
                + torch.cos(
                    math.pi - sample_margin
                )
            )

            phi = torch.where(
                cosine > th,
                phi,
                cosine - mmm
            )


        # =====================================================
        # 6. Only replace target-class logit
        # =====================================================

        one_hot = input.new_zeros(
            cosine.size()
        )

        one_hot.scatter_(
            1,
            label.view(-1, 1).long(),
            1.0
        )


        output = (
            one_hot * phi
            +
            (1.0 - one_hot) * cosine
        )


        # =====================================================
        # 7. Standard scale
        # =====================================================

        output *= self.scale

        return output


    def extra_repr(self):

        return (
            'in_features={}, '
            'out_features={}, '
            'scale={}, '
            'margin={}, '
            'easy_margin={}, '
            'margin_std={}, '
            'reference_margin={}, '
            'max_margin={}'
        ).format(
            self.in_features,
            self.out_features,
            self.scale,
            self.margin,
            self.easy_margin,
            self.margin_std,
            self.reference_margin,
            self.max_margin
        )


class ConfidenceCalibratedArcMarginProduct(nn.Module):

    def __init__(self, in_features, out_features, scale=32.0, margin=0.2,
                 easy_margin=False, confidence_alpha=0.1, confidence_tau=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.confidence_alpha = confidence_alpha
        self.confidence_tau = confidence_tau
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def update(self, margin):
        self.margin = margin

    def forward(self, input, label):

        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.clamp(-1+1e-7, 1-1e-7)

        target = cosine.gather(1, label.view(-1, 1))
        sine = torch.sqrt(torch.clamp(1-target.pow(2), min=1e-7))

        phi = target * math.cos(self.margin) - sine * math.sin(self.margin)

        if self.easy_margin:
            phi = torch.where(target > 0, phi, target)
        else:
            th = math.cos(math.pi-self.margin)
            mm = 1 + math.cos(math.pi-self.margin)
            phi = torch.where(target > th, phi, target-mm)

        output = cosine.clone()
        output.scatter_(1, label.view(-1,1), phi)

        with torch.no_grad():
            mask = torch.zeros_like(cosine)
            mask.scatter_(1, label.view(-1,1), 1)

            hard_neg = cosine.masked_fill(mask.bool(), -2).max(dim=1).values
            confidence = torch.sigmoid((target.squeeze(1)-hard_neg)/self.confidence_tau)

        output.scatter_add_(1, label.view(-1,1),
                            (self.confidence_alpha * confidence).view(-1,1))

        return output * self.scale

class Linear(nn.Module):
    """
    The linear transform for simple softmax loss
    """

    def __init__(self, emb_dim=512, class_num=1000):
        super(Linear, self).__init__()

        self.trans = nn.Sequential(nn.BatchNorm1d(emb_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(emb_dim, class_num))

    def forward(self, input, label):
        out = self.trans(input)
        return out


if __name__ == '__main__':
    #    projection = ArcMarginProduct(100,
    #                                  200,
    #                                  scale=32.0,
    #                                  margin=0.2,
    #                                  easy_margin=False)
    #
    #    print(hasattr(projection, 'update_mar'))
    projection = ArcMarginProduct_intertopk_subcenter(100,
                                                      200,
                                                      scale=32.0,
                                                      margin=0.0,
                                                      easy_margin=False,
                                                      K=3,
                                                      mp=0.06,
                                                      k_top=5)
    print(hasattr(projection, 'update'))
    projection.update(0.2)
    print(projection)
    embed = torch.randn(16, 100)
    label = torch.randint(200, (16, ))
    out = projection(embed, label)
    print(out.size())

    # for name, param in projection.named_parameters():
    #     print(name)
    #     print(param.shape)
