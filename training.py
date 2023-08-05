from torch import optim, utils
from hiera import mae_hiera_tiny_224
from hiera.hiera_mae import MaskedAutoencoderHiera
import torchvision
from utils import FolderDataset


path = "/home/yandex/MLFH2023/giladd/hiera/datasets/**/"


def main(model: MaskedAutoencoderHiera):
    dataset = FolderDataset(
        path=path,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader = utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=0
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    i = 0
    for batch in dataloader:
        loss = model.forward(batch)[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    model.eval()


if __name__ == "__main__":
    model: MaskedAutoencoderHiera = mae_hiera_tiny_224()
    main(model)
