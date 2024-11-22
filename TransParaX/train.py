import torch
from torch.utils.data import DataLoader
from loss import ParameterExtractionLoss
import numpy as np
from torch.utils.data import TensorDataset

def train_model(model, data_loader, num_epochs=300, device='cuda'):
    criterion = ParameterExtractionLoss(param_weights=torch.ones(14).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for curves, params in data_loader:
            curves, params = curves.to(device), params.to(device)
            
            optimizer.zero_grad()
            
            pred_dict = model(curves)
            
            loss, loss_dict = criterion(pred_dict, params)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss_dict["mse_loss"]
        epoch_loss /= len(data_loader)
        losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
    torch.save(model.state_dict(), 'complete_modelv2.pth')
    return losses

def train_model_val(model, data_loader, val_loader, num_epochs=300, patience=100,device='cuda'):

    criterion = ParameterExtractionLoss(
        param_weights=torch.ones(14).to(device)  
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []  
    
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        for curves, params in data_loader:
            curves, params = curves.to(device), params.to(device)  
            
            optimizer.zero_grad()
            
            # forward
            pred_dict = model(curves)
            
            # loss
            loss, loss_dict = criterion(pred_dict, params)
            
            # backward
            loss.backward()
            optimizer.step()
            
            #epoch_loss += loss.item()
            epoch_loss += loss_dict["mse_loss"]
        epoch_loss /= len(data_loader)
        losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        model.eval()
        loss_val = []
        val_losses = []
        with torch.no_grad():
            for batch_curves, batch_params in val_loader:
                batch_curves,batch_params = batch_curves.to(device),batch_params.to(device)
                pred_dict = model(batch_curves)
                val_loss = criterion(pred_dict['mean'], batch_params)
                val_losses.append(val_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        loss_val.append(avg_val_loss)
        print(f"val_loss: {avg_val_loss}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_modelv2.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return losses,loss_val  

def train_with_semi_supervision(model, labeled_loader, unlabeled_data, 
                                num_epochs=300, initial_data_size=1000, 
                                confidence_threshold=0.8, batch_size=32,device='cuda'):
    '''
    use_age
    # # Now use a small labeled dataset for initial training, and pass the remaining as unlabeled data
    # labeled_indices = np.random.choice(len(dataX), size=initial_data_size, replace=False)
    # labeled_dataX = dataX[labeled_indices]
    # labeled_dataY = dataY[labeled_indices]
    
    # unlabeled_dataX = np.delete(dataX, labeled_indices, axis=0)
    # unlabeled_dataY = np.delete(dataY, labeled_indices, axis=0)
    
    # # Prepare PyTorch DataLoader
    # labeled_dataset = TensorDataset(torch.tensor(labeled_dataX, dtype=torch.float32), 
    #                                 torch.tensor(labeled_dataY, dtype=torch.float32))
    # labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)
    
    # # Initial training with semi-supervision
    # losses = train_with_semi_supervision(model, labeled_loader, unlabeled_dataX, test_loader)
    '''
    # Optimizer and criterion setup
    criterion = ParameterExtractionLoss(param_weights=torch.ones(14).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Initialize labeled dataset
    labeled_data_size = initial_data_size
    labeled_indices = np.random.choice(len(unlabeled_data), labeled_data_size, replace=False)
    labeled_data = unlabeled_data[labeled_indices]
    remaining_unlabeled_data = np.delete(unlabeled_data, labeled_indices, axis=0)
    
    best_val_loss = float('inf')
    patience_counter = 0
    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        # Training on labeled data
        for curves, params in labeled_loader:
            curves, params = curves.to(device), params.to(device)
            optimizer.zero_grad()
            pred_dict = model(curves)
            loss, loss_dict = criterion(pred_dict, params)
            loss.backward()
            optimizer.step()
            epoch_loss += loss_dict["mse_loss"]

        epoch_loss /= len(labeled_loader)
        losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Labeled Loss: {epoch_loss:.4f}')

        # Generate pseudo-labels for unlabeled data if confidence is high
        model.eval()
        pseudo_labels = []
        pseudo_inputs = []

        with torch.no_grad():
            for i in range(0, len(remaining_unlabeled_data), batch_size):
                batch = remaining_unlabeled_data[i:i+batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
                pred_dict = model(batch_tensor)
                
                # Select samples with high confidence (low variance)
                confidence = 1.96 * torch.sqrt(pred_dict['variance']).cpu().numpy()
                mask = (confidence < confidence_threshold).all(axis=1)
                
                pseudo_labels.extend(pred_dict['mean'][mask].cpu().numpy())
                pseudo_inputs.extend(batch[mask])

        # Add pseudo-labeled data to labeled dataset
        if pseudo_inputs:
            pseudo_inputs = torch.tensor(np.array(pseudo_inputs), dtype=torch.float32)
            pseudo_labels = torch.tensor(np.array(pseudo_labels), dtype=torch.float32)
            labeled_data = torch.cat((labeled_data, TensorDataset(pseudo_inputs, pseudo_labels)), dim=0)
            labeled_loader = DataLoader(labeled_data, batch_size=batch_size, shuffle=True)
            print(f'Added {len(pseudo_inputs)} pseudo-labeled samples to labeled dataset.')

    return losses

# Transfer learning with adapter
def train_model_with_adapter(model, data_loader, num_epochs=300,device='cuda'):
    # 首先冻结基础模型参数
    model.load_state_dict(torch.load("best_model.pth"))
    model.freeze_base_model()
    
    criterion = ParameterExtractionLoss(
        param_weights=torch.ones(14).to(device)
    )
    # use small learning rate for adapter layers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for curves, params in data_loader:
            curves, params = curves.to(device), params.to(device)
            
            optimizer.zero_grad()
            pred_dict = model(curves)
            loss, loss_dict = criterion(pred_dict, params)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss_dict["mse_loss"]
            
        epoch_loss /= len(data_loader)
        losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')