if __name__ == "__main__":
    from model import PitchPredictorMLP
    from dataset import AudioFeatureDataset, AudioFeatureCollator, train_val_split

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    model = PitchPredictorMLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    dataset = AudioFeatureDataset('testmulti_pp/filelist.list')
    collator = AudioFeatureCollator()
    train_dataset, val_dataset = train_val_split(dataset, 0.2)
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    # Basically: Does mean of HuBERT features contain any pitch information?
    EPOCHS = 200
    for epoch in range(EPOCHS):
        for batch in tqdm(train_dataloader, desc="Training"):
            optimizer.zero_grad()
            outputs = model(batch['features'].mean(dim=1))

            norm_constant = 300
            f0_mask = batch['f0'] != 0
            sum_f0 = torch.sum(batch['f0'].to(torch.float32) * f0_mask, dim=1)
            mean_f0_nonzero = sum_f0 / torch.sum(f0_mask, dim=1) / norm_constant
            # (get f0 in more sane range for loss)

            loss = criterion(outputs, mean_f0_nonzero)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item()}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Validation")):
                outputs = model(batch['features'].mean(dim=1))

                norm_constant = 300
                f0_mask = batch['f0'] != 0
                sum_f0 = torch.sum(batch['f0'].to(torch.float32) * f0_mask, dim=1)
                mean_f0_nonzero = sum_f0 / torch.sum(f0_mask, dim=1) / norm_constant
                # (get f0 in more sane range for loss)

                loss = criterion(outputs, mean_f0_nonzero)  
                if batch_idx % 10 == 0:
                    print(f"Predicted: {outputs[0].item() * norm_constant}, Actual: {mean_f0_nonzero[0].item() * norm_constant}")
            print(f"Validation Loss: {loss.item()}")