import Datasets 
#Instantiate Datasets object and get k cifar10 images
d = Datasets.Datasets()
#Array of queried images
print(d.getCifar10Images(5))