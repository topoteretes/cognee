## Creating the EC2 Instance

Create an EC2 Instance with the 

`Ubuntu Image`

Many instance types will work, we used:

`m7a.2xlarge` # more than 8 parallel processes doesn't seem to speed up overall process. Maybe to do with docker parallelism?

DON'T FORGET TO ADD

`500 GB storage`

Or the evaluation run will run out of space

Add a key pair login where you have access to the corresponding key file (*.pem)

## Accessing your instance and setup

To ssh into the instance, you have to save your key pair file (*.pem) to an appropriate location, such as ~/.aws. After launching the instance, you can access the Instance Summary, and retrieve "Public IPv4 DNS" address. Then run

`ssh -i PATH_TO_KEY ubuntu@IPv4ADDRESS`

to gain command line access to the instance.

To copy your current state of cognee, go to the folder that contains "cognee" on your local machine, zip it to cognee.zip and run:

`zip -r cognee.zip cognee`
`scp -i PATH_TO_KEY cognee.zip ubuntu@IPv4ADDRESS:cognee.zip`

And unzip cognee.zip in your SSH session:

`sudo apt install unzip`
`unzip cognee.zip`

Then run:
`cd cognee`
`source evals/cloud/setup_ubuntu_instance.sh`

`sudo usermod -aG docker $USER`

disconnect, and reconnect. 

Confirm that `ubuntu` has been added to the docker user group with

`groups | grep docker`

## Running SWE-bench

Then enter a `screen` and activate the virtual env

`screen`
`source venv/bin/activate`

then, from cognee, you can run swe_bench:

`cd cognee`

`python evals/eval_swe_bench.py --cognee_off --max_workers=N_CPUS`

Building the environment images should take roughly 17 minutes

If the virtual env wasn't set up correctly for some reason, just run the last few lines of `setup_ubuntu_instance.sh` manually