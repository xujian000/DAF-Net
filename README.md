# DAF

### Env

```
# create virtual environment
conda create -n DAF python=3.8.10
conda activate DAF
pip install -r requirements.txt
# pytorch  : pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### 🏄 Testing

```
python test.py
```

### 🏊 Training

**1. Data Preparation**

Download the MSRS dataset from [this link](https://github.com/Linfeng-Tang/MSRS) and place it in the folder `'./MSRS_train/'`.

**2. Pre-Processing**

Run

```
python prepare_data.py
```

and the processed training dataset is in `'./data/MSRS_train_imgsize_128_stride_200.h5'`.

**3. Training**

Run

```
python train.py
```

and the trained model is available in `'./models/'`.
