import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import torchaudio  # Ensure torchaudio is installed

class InTheWildDataset(Dataset):
    def __init__(self, csv_path, data_path, transform=None):
        """
        Args:
            csv_path (str): Path to the CSV metadata file.
            data_path (str): Path to the directory containing audio files.
            transform (callable, optional): Optional transform to apply to the samples.
        """
        self.metadata = pd.read_csv(csv_path)
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_file = os.path.join(self.data_path, self.metadata.iloc[idx, 0])
        speaker = self.metadata.iloc[idx, 1]
        label = self.metadata.iloc[idx, 2]

        # Convert label to tensor (0 = bona-fide, 1 = spoof)
        label_tensor = torch.tensor(1 if label == 'spoof' else 0, dtype=torch.long)

        # Load the audio file using torchaudio
        waveform, sample_rate = torchaudio.load(audio_file)

        # Apply transforms if any
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label_tensor
