dir=$(pwd)
sudo docker build -t pytorch -f docker/pytorch-no-cuda/Dockerfile . && sudo docker run -it -p 6006:6006 -v $dir/mlis:/mlinseconds-find-me -w /mlinseconds-find-me --name pytorch --rm pytorch
sudo docker run -it -p 6006:6006 -v $dir/mlis:/mlinseconds-find-me -w /mlinseconds-find-me --name pytorch --rm pytorch
