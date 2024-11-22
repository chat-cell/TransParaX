import torch
from TransParaX.model import DeviceParameterExtractor
from TransParaX.data import load_data, create_dataloaders
from TransParaX.train import train_model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataX, dataY, mean, std, meany, stdy = load_data("dmodel/data.npy", "dmodel/params.npy")
    
    train_loader, test_loader = create_dataloaders(dataX, dataY)

    model = DeviceParameterExtractor().to(device)

    losses = train_model(model, train_loader, num_epochs=300, device=device)

if __name__ == "__main__":
    main()
