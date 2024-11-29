
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo docker run hello-world

sudo apt install unzip

sudo apt-get install python3-virtualenv

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

sudo apt install python3.11

virtualenv venv --python=python3.11

source venv/bin/activate

pip install poetry

poetry install

pip install swebench transformers sentencepiece

groups | grep docker

python evals/eval_swe_bench.py --cognee_off

sudo usermod -aG docker $USER

