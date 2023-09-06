# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any

import torch as T
from torch import Tensor, nn

__all__ = [
    "AlexNet",
    "alexnet",
]


class ReLU2(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.bound = 2

    def forward(self, x):
        return T.clamp_max(T.relu(x), self.bound)

    def __repr__(self):
        return "ReLU6"


class NNAlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.lookup = [[3, 6, 8, 10], [1, 4, 5]]

        self.features = nn.Sequential()
        self.features_list = [
            None,
            lambda: nn.ReLU(inplace=True),
            lambda: nn.MaxPool2d(kernel_size=3, stride=2),
            None,
            lambda: ReLU2(),
            lambda: nn.MaxPool2d(kernel_size=3, stride=2),
            None,
            lambda: ReLU2(),
            None,
            lambda: ReLU2(),
            None,
            lambda: ReLU2(),
            lambda: nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential()
        self.classifier_list = [
            lambda: nn.Dropout(p=dropout),
            None,
            lambda: ReLU2(),
            lambda: nn.Dropout(p=dropout),
            None,
            lambda: ReLU2(),
            lambda: nn.Linear(4096, num_classes),
        ]

        self.counter = 0

    def add_layer(self, layer):
        if self.counter < 5:
            i = len(self.features)
            while self.features_list[i] is not None:
                self.feature.append(self.feature_list[i]())
                i += 1
            self.features[i].append(layer)
        else:
            i = len(self.classifier)
            while self.classifier_list[i] is not None:
                self.classifier.append(self.feature_list[i]())
                i += 1
            self.classifier[i].append(layer)

        self.counter += 1

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = T.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 dropout: float = 0.5,
                 alpha=None) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            ReLU2(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            ReLU2(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            ReLU2(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            ReLU2(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU2(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            ReLU2(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            ReLU2(),
            nn.Linear(4096, num_classes),
        )

        self.alpha = T.tensor(alpha)

        self.layers = []
        for layer in self.children():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                self.layers.append(layer)

    def get_nn_net(self):
        return NNAlexNet()

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = T.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs: Any) -> AlexNet:
    model = AlexNet(**kwargs)

    return model
