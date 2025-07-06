flowchart TD
    %% Entry & Config
    A0([Start: main()])
    A0 --> A1[Initialize Config: Paths, Classes, Hyperparams]
    A1 --> A2[Set Random Seed]
    A2 --> A3[Setup Logging: Console & File]
    A3 --> A4[Save Config as JSON]

    %% Data
    A4 --> B1[Create Datasets: Train, Val, Test]
    B1 --> B2[Apply Data Augmentations]
    B2 --> B2a[Train Transforms: Crop, Flip, Jitter, Blur, Rotate, Erase]
    B2 --> B2b[Test Transforms: Resize, CenterCrop, Normalize]
    B2 --> B3[TTA Transforms: hflip, vflip]

    %% DataLoader
    B1 --> C1[Create WeightedRandomSampler]
    C1 --> C2[Build DataLoaders]

    %% Model & Optimizer
    C2 --> D1[Create Model (SwinV2)]
    D1 --> D2[Send to DEVICE]
    D2 --> D3[Setup Optimizer (AdamW)]
    D3 --> D4[Setup Scheduler (CosineAnnealingLR)]
    D4 --> D5[Setup GradScaler (Mixed Precision)]
    D5 --> D6[Setup Losses: CrossEntropy, SupCon, Mixup/Cutmix]

    %% Training Loop
    D6 --> E1{For each Epoch}
    E1 --> E2[Train Loop]
    E2 --> E3[For each batch: Mixup?]
    E3 --> E4[Forward Pass & Compute Loss]
    E4 --> E5[Backward, GradAccum, Clip]
    E5 --> E6[Optimizer Step]
    E2 --> E7[Track Metrics]
    E1 --> F1[Validation]
    F1 --> F2[Compute Metrics & Save]

    %% Checkpointing
    F2 --> G1{Val AUC > Best?}
    G1 -- Yes --> G2[Save Best Model]
    G1 -- No --> G3[Early Stop Check]
    G3 -- Yes --> G4[Break Loop]
    G3 -- No --> E1

    %% Testing / TTA
    G2 & G4 --> H1[Load Best Model]
    H1 --> H2{TTA Enabled?}
    H2 -- Yes --> H3[Run TTA Inference]
    H2 -- No --> H4[Standard Test Inference]
    H3 & H4 --> H5[Compute & Save Metrics/Plots]

    %% Cleanup
    H5 --> I1[GC Collect, CUDA Empty Cache]
    I1 --> J([End])
