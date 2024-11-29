Create an EC2 Instance with the 

`Ubuntu Image`

Many instance types will work, we used:

`m7a.2xlarge` # more than 8 parallel processes doesn't seem to speed up overall process. Maybe to do with docker parallelism?

DON'T FORGET TO ADD

`500 GB storage`

Or the evaluation run will run out of space

--------------------------------------------------------

Then ssh into the instance, run

source evals/cloud/setup_ubuntu_instance.sh

sudo usermod -aG docker $USER

disconnect, and reconnect. 

Then enter a `screen` and activate the virtual env

screen
source venv/bin/activate

then, from cognee, you can run swe_bench:

python evals/eval_swe_bench --cognee_off --max_workers=N_CPUS

Building the environment images takes roughly 17 minutes