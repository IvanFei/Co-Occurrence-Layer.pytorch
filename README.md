## Note 
#### This repo is pytorch implement for "Co-Occurrence Neural Network"

## Co-Occurrence Convolution
<div align="center">
<img src="docs/imgs/Co-Occurrence%20Convolution.png" width = "500" height = "400"/>
</div>

## Usage

- clone the repo
```bash
git clone git@github.com:IvanFei/Co-Occurrence-Layer.pytorch.git
cd Co-Occurrence-Layer.pytorch
```

- train the Co-Occurrence Network with gpu
```bash
python main.py --model_name Conn --model_type Conn --mode train --gpu_id 0
```

- train the Co-Occurrence Network with cpu
```bash
python main.py --model_name Conn --model_type Conn --mode train --gpu_id -1
```

## Co-Occurrence Conv Test Result
<div align="center">
<img src="docs/imgs/input%20matrix.png" width = "400" height = "300"/> 
<img src="docs/imgs/input%20index.png" width = "400" height = "300"/>

<img src="docs/imgs/co%20occurrence%20matrix.png" width = "400" height = "300"/>
<img src="docs/imgs/Co%20Occurrence%20output.png" width = "400" height = "300"/>
</div>


