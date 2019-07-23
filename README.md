# train cityscapes-dataset using UNet model

``` bash
git clone https://github.com/anhdhbn/cityscapes
cd cityscapes
chmod 777 download_data.sh
./download_data.sh

python preprocess_data.py

python train.py

```
