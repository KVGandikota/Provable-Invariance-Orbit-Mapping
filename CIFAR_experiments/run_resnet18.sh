##std. augment orbit mapping
for i in 1 2 3 4 5
do
  python train_test_cifar.py --gradalign --net Resnet18 --lr 0.1 --run $i
done

##rot-augment orbit mapping
for i in 1 2 3 4 5
do
  python train_test_cifar.py --gradalign --net Resnet18 --augment 'rot-default' --lr 0.1 --run $i
done

##RA 
for i in 1 2 3 4 5
do
  python train_test_cifar.py --net Resnet18 --augment 'rot-default' --lr 0.1 --run $i
done

##Std. train
for i in 1 2 3 4 5
do
  python train_test_cifar.py --net Resnet18  --lr 0.1 --run $i
done
