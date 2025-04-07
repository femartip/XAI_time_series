import torch

# Classifier for time series classification
class ConvClassifier(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvClassifier, self).__init__()

        self.num_classes = num_classes
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8)
        self.batch_norm1 = torch.nn.BatchNorm1d(128)
        
        self.conv2 = torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5)
        self.batch_norm2 = torch.nn.BatchNorm1d(256)

        self.conv3 = torch.nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3)
        self.batch_norm3 = torch.nn.BatchNorm1d(128)

        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)

        self.fc = torch.nn.Linear(128, num_classes)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.relu(self.batch_norm3(self.conv3(x)))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    


        
        