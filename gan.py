import torch
import torchvision
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
from torch import nn

from torch.utils.data import Dataset
import os
from PIL import Image
from torch.utils.data import DataLoader

torch.manual_seed(111)

device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Инициализация датасета.

        :param data_dir: Путь к директории с изображениями.
        :param transform: Преобразования, которые нужно применить к изображениям.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.img_names = os.listdir(data_dir)  # Список имен файлов в директории

        # Проверка на соответствие количества изображений и меток
        self.labels = [self.get_label(name) for name in self.img_names]
        if len(self.img_names) != len(self.labels):
            print("Warning: Number of images does not match number of labels.")

    def __len__(self):
        """Возвращает количество изображений в датасете."""
        return len(self.img_names)

    def __getitem__(self, idx):
        """Получает изображение и метку по индексу."""
        img_name = self.img_names[idx]
        img_path = os.path.join(self.data_dir, img_name)

        # Загружаем изображение
        image = Image.open(img_path).convert('L')  # 'L' для градаций серого

        # Получаем метку
        label = self.get_label(img_name)

        # Применяем преобразования, если они указаны
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_label(self, img_name):
        """Метод для получения метки из имени файла."""
        try:
            label_str = img_name.split('_')[1]  # Предположим, что метка в имени файла
            label = int(label_str.split('.')[0])  # Удаляем расширение и преобразуем в int
            return label
        except (IndexError, ValueError) as e:
            print(f"Error extracting label from {img_name}: {e}")
            return -1  # Значение по умолчанию в случае ошибки


# Определяем преобразования (например, преобразование в тензор и нормализация)
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Убедитесь, что размер 28x28
    transforms.ToTensor(),  # Преобразуем в тензор
    transforms.Normalize((0.5,), (0.5,))  # Нормализация для градаций серого
])

# Создаем экземпляр вашего датасета
data_dir = '/content/data'  # Укажите путь к вашим изображениям
custom_set = CustomDataset(data_dir=data_dir, transform=transform)


# Создаем DataLoader
batch_size = 16
data_loader = DataLoader(custom_set, batch_size, shuffle=True)

# Итерация по DataLoader
for images, labels in data_loader:
    print(images.shape)  # Должно быть [batch_size, 1, 28, 28]
    print(labels)        # Метки для текущего пакета
    break  # Прерываем после первого пакета для проверки

real_samples, mnist_labels = next(iter(data_loader))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(real_samples[i].reshape(28, 28))
    plt.xticks([])
    plt.yticks([])


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output
    

discriminator = Discriminator().to(device=device)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output


generator = Generator().to(device=device)

lr = 0.0001
num_epochs = 50
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)


for epoch in range(num_epochs):
    #for n, (real_samples, mnist_labels) in enumerate(data_loader):
    #    print(f"Batch {n}: real_samples size: {real_samples.size()}, mnist_labels size: {mnist_labels.size()}")
    for n, (real_samples, mnist_labels) in enumerate(data_loader):
        # Данные для тренировки дискриминатора
        real_samples = real_samples.to(device=device)
        real_samples_labels = torch.ones((batch_size, 1)).to(
            device=device)
        latent_space_samples = torch.randn((batch_size, 100)).to(
            device=device)
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1)).to(
            device=device)
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels))

        # Обучение дискриминатора
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)

        #all_sample_labels обрезан
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Данные для обучения генератора
        latent_space_samples = torch.randn((batch_size, 100)).to(
            device=device)

        # Обучение генератора
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

        # Показываем loss
        if n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")


latent_space_samples = torch.randn(batch_size, 100).to(device=device)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.cpu().detach()

for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(generated_samples[i].reshape(28, 28))
    #plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])