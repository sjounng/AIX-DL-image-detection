"""
딥러닝 모델 정의

이 모듈은 AI 이미지 판별을 위한 다양한 딥러닝 모델을 제공합니다.

모델 종류:
    - SimpleCNN: 베이스라인 CNN 모델
    - ResNet50: 전이학습 모델
    - EfficientNetB0: 전이학습 모델
    - VGG16: 전이학습 모델
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """
    간단한 CNN 베이스라인 모델

    구조:
        Conv2D (64) -> ReLU -> MaxPool
        Conv2D (128) -> ReLU -> MaxPool
        Conv2D (256) -> ReLU -> MaxPool
        Flatten -> Dense (512) -> ReLU -> Dropout
        Dense (2) -> Softmax

    Args:
        num_classes (int): 출력 클래스 개수 (기본값: 2)
        dropout (float): Dropout 비율 (기본값: 0.5)
    """

    def __init__(self, num_classes=2, dropout=0.5):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv Block 1: 3 -> 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112

            # Conv Block 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56

            # Conv Block 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28

            # Conv Block 4: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, 3, 224, 224]

        Returns:
            Output tensor [batch_size, num_classes]
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def get_resnet50(num_classes=2, pretrained=True):
    """
    ResNet50 전이학습 모델 생성

    Args:
        num_classes (int): 출력 클래스 개수
        pretrained (bool): ImageNet 사전학습 가중치 사용 여부

    Returns:
        ResNet50 모델
    """
    # 사전학습된 ResNet50 로드
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    # 마지막 FC layer 교체 (1000 -> num_classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def get_efficientnet_b0(num_classes=2, pretrained=True):
    """
    EfficientNet-B0 전이학습 모델 생성

    Args:
        num_classes (int): 출력 클래스 개수
        pretrained (bool): ImageNet 사전학습 가중치 사용 여부

    Returns:
        EfficientNet-B0 모델
    """
    # 사전학습된 EfficientNet-B0 로드
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)

    # 마지막 classifier layer 교체
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    return model


def get_vgg16(num_classes=2, pretrained=True):
    """
    VGG16 전이학습 모델 생성

    Args:
        num_classes (int): 출력 클래스 개수
        pretrained (bool): ImageNet 사전학습 가중치 사용 여부

    Returns:
        VGG16 모델
    """
    # 사전학습된 VGG16 로드
    if pretrained:
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
    else:
        model = models.vgg16(weights=None)

    # 마지막 classifier layer 교체 (1000 -> num_classes)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)

    return model


def get_model(model_name, num_classes=2, pretrained=True):
    """
    모델 이름으로 모델 생성

    Args:
        model_name (str): 모델 이름
            - 'simple_cnn': SimpleCNN
            - 'resnet50': ResNet50
            - 'efficientnet_b0': EfficientNet-B0
            - 'vgg16': VGG16
        num_classes (int): 출력 클래스 개수
        pretrained (bool): 사전학습 가중치 사용 여부 (SimpleCNN 제외)

    Returns:
        PyTorch 모델

    Raises:
        ValueError: 지원하지 않는 모델 이름
    """
    model_name = model_name.lower()

    if model_name == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes)
    elif model_name == 'resnet50':
        return get_resnet50(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'efficientnet_b0':
        return get_efficientnet_b0(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'vgg16':
        return get_vgg16(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Choose from: simple_cnn, resnet50, efficientnet_b0, vgg16")


def count_parameters(model):
    """
    모델의 파라미터 개수 계산

    Args:
        model: PyTorch 모델

    Returns:
        dict: 전체/학습가능 파라미터 개수
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params
    }


def test_models():
    """
    모든 모델 테스트 함수

    각 모델의 입출력 shape과 파라미터 개수를 확인합니다.
    """
    print("="*60)
    print("모델 테스트")
    print("="*60)

    # 테스트용 입력 (batch_size=4, channels=3, height=224, width=224)
    dummy_input = torch.randn(4, 3, 224, 224)

    models_to_test = [
        'simple_cnn',
        'resnet50',
        'efficientnet_b0',
        'vgg16'
    ]

    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"{model_name.upper()}")
        print('='*60)

        # 모델 생성 (사전학습 가중치 없이 - 테스트 속도를 위해)
        if model_name == 'simple_cnn':
            model = get_model(model_name)
        else:
            model = get_model(model_name, pretrained=False)

        model.eval()

        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)

        # 파라미터 개수
        params = count_parameters(model)

        # 결과 출력
        print(f"\nInput shape:  {tuple(dummy_input.shape)}")
        print(f"Output shape: {tuple(output.shape)}")
        print(f"\nParameters:")
        print(f"  Total:     {params['total']:,}")
        print(f"  Trainable: {params['trainable']:,}")

        # 메모리 사용량 (대략적)
        memory_mb = params['total'] * 4 / (1024 ** 2)  # float32 = 4 bytes
        print(f"  Memory:    ~{memory_mb:.1f} MB")

    print("\n" + "="*60)
    print("✅ 모든 모델 테스트 완료!")
    print("="*60)

    # 모델 비교 테이블
    print("\n" + "="*60)
    print("모델 비교 (사전학습 가중치 사용 시)")
    print("="*60)
    print(f"\n{'Model':<20} {'Parameters':<15} {'Memory (MB)':<12}")
    print("-" * 60)

    comparison_data = [
        ('SimpleCNN', '~10M', '~40'),
        ('ResNet50', '~25M', '~100'),
        ('EfficientNet-B0', '~5M', '~20'),
        ('VGG16', '~138M', '~550'),
    ]

    for name, params, memory in comparison_data:
        print(f"{name:<20} {params:<15} {memory:<12}")

    print("\n권장 사항:")
    print("  - 빠른 실험: EfficientNet-B0")
    print("  - 높은 성능: ResNet50")
    print("  - 베이스라인: SimpleCNN")
    print("="*60)


if __name__ == "__main__":
    # 테스트 실행
    test_models()
