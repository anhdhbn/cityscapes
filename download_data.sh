# import os

# data_dir = "/content/cityscapes/data/"

# if os.path.isdir( data_dir + "gtFine/") and  os.path.isdir(data_dir + "leftImg8bit/"):
#     pass
# else:
#   ! 
#   ! 
#   ! unzip -qq -o {data_dir}gtFine_trainvaltest.zip -d {data_dir}
#   ! 
#   ! unzip -qq -o {data_dir}leftImg8bit_trainvaltest.zip -d {data_dir}


wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=anhdhbn&password=Honganh99&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1 -P data
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3 -P data

cd data
unzip -qq gtFine_trainvaltest.zip
unzip -qq leftImg8bit_trainvaltest.zip