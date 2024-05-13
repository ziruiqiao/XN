import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class BookshelfDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['crop_path']  # Assuming image path is last column
        image = Image.open(img_path).convert('RGB')
        label = int(self.dataframe.iloc[idx]['Label'])

        if self.transform:
            image = self.transform(image)

        return image, label


class CustomCNNModel:
    def __init__(self, model):
        self.model = model
    @classmethod
    def load_model(self, path):
        model = models.resnet18(pretrained=False)

        model_path = path
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return CustomCNNModel(model)

    def predict(self, x):
        with torch.no.grad():
            preds = self.model(x)
        return preds

    def predict_proba(self, x):
        with torch.no.grad():
            probs = self.model.predict_proba(x)[:, 1]
        return probs
