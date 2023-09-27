# pffrocd
Privacy-Friendly Face Recognition On Constrained Devices

## Test explanation

### Devices needed

Three entities are involved in the testing: a _master_ device, _server_ (simulating a drone) and _client_ (simulating a mobile device). They should be located in the same network so that _master_ can `ssh` into _server_ and _client_ and run SFE on them.

![Untitled Diagram(18) drawio](https://github.com/Musialke/pffrocd/assets/26610983/128361a4-b817-42ac-a857-e592e41556b4)


The _master_ device does not need to be a very powerful machine, as it only orchestrates the operations. Its specifications do not influence test results. A Raspberry Pi 2 was performing just fine.

I was running:

_master_: RPi Model 3 B (aarch64) DietPi v8.20.1 Debian GNU/Linux 12 (bookworm)


_client_ and _server_:
RPi Model 3 B+
Raspbian OS
Operating System: Debian GNU/Linux 11 (bullseye)
          Kernel: Linux 6.1.21-v8+
    Architecture: arm64


### Running tests

First, follow the [setup guide](#setup-guide). Then execute the main script `pyscripts/master.py` at the _master_ device. It handles the communication with the _server_ and _client_.

The _master_ device:
- reads the config file
- prepares images from the image database
- creates face embedding shares and sends them to _server_ and _client_
- orchestrates SFE execution between _server_ and _client_
- saves results

There is logging:
INFO level logging to stdout and DEBUG level logging to a file in `log/`.

The results are saved in a `.csv` file as a `pandas` Dataframe in the `dfs/` folder. For the format of the saved data, see `pffrocd.columns` in `pyscripts/pffrocd.py`. Also, the Jupyter Notebooks in `plotting/` and `results/` can give an overview of how to visualize the data.

### Testing flow

The testing flow is as follows:

**Master**

1. Reads config file
2. Tests bandwidth between _sever_ and _client_ (by remotely executing iperf3 tests on both devices)
3. Prepares the database images (choose appropriate people from the DB and their images)
4. Sets image x of person p as the reference image (the image stored at the Service Provider)
5. Sends share of x to _client_ and _server_
6. Runs tests for other images of p, namely for each image i:
7. Makes the _server_ extract embedding of i
8. Sends embeddings to _client_ and _server_
9. Runs SFE on _client_ and _server_
10. If indicated, reruns SFE this time gathering energy data from Powertop
11. Saves results

![image](https://github.com/Musialke/pffrocd/assets/26610983/e0843c66-283b-4aea-b536-fe309f1481fd)

## Setup Guide:

**For all three devices**

1. Install required packages:

```sh
sudo apt update && sudo apt install time python3 python3-venv iperf3 g++ make cmake libgmp-dev libssl-dev libboost-all-dev ffmpeg libsm6 libxext6 git powertop -y
```

2. Generate SSH keys and add them as deploy keys to the git repo (to be able to clone the repo)

```sh
ssh-keygen
```

3. Clone the repo and cd into it

```sh
git clone git@github.com:Musialke/pffrocd.git
cd pffrocd
```

**For _server_ and _client_:**

4. Create the ABY build directory
```sh
mkdir ABY/build/ && cd ABY/build/
```

5. Use CMake to configure the build (example applications on by default):
```sh
cmake ..
```

6. Call `make` in the build directory. You can find the build executables and libraries in the `bin/` and `lib/` directories, respectively.
```sh
make
```

7. To be able to run a process with higher priority, modify limits.conf as explained here: https://unix.stackexchange.com/a/358332

8. Calibrate Powertop to get power estimate readings

```sh
sudo powertop --calibrate
```

This takes a while, turns peripherals on and off and reboots the system.

**ADDITIONALLY for _master_ and _server_:**

Since the _server_ and _master_ need to extract embeddings, they need the database of pictures and Python.

9. Change the directory back to the repo root folder and unpack the picture database:
```sh
cat lfw.tgz.parta* | tar -xzv
```

10. Create a new virtual environment, activate it and install the required packages
```sh
python3 -m venv env
. env/bin/activate
pip install -vr requirements.txt
```

11. Copy the SFace weights where deepface can find them:
```sh
mkdir -p ~/.deepface/weights/ && cp face_recognition_sface_2021dec.onnx ~/.deepface/weights/
```
**ADDITIONALLY for _master_**

You need to specify config options, and _master_ needs to be able to ssh into _server_ and _client_. 

12. Rename the `config.ini.example` file to `config.ini` and modify it accordingly

13. Copy the SSH keys to the _server_ and _client_ using ssh-copy-id

```sh
ssh-copy-id user@ip_address
```
**All done!**
You can now run the main script in the background on the master machine
```sh
nohup python3 pyscripts/master.py </dev/null >/dev/null 2>&1 &
```

Follow the logs to know what stage the program is at:
```sh
tail -f log/<logfile>
```

The logs are saved in the `log/` directory and the test results are appended to a csv file in `dfs/` after running all tests for one person.


### Possible errors and solutions:

`ImportError: libGL.so.1: cannot open shared object file: No such file or directory`
Fix:
```sh
sudo apt update && sudo apt install ffmpeg libsm6 libxext6  -y
```

`v2.error: OpenCV(4.7.0) /io/opencv/modules/dnn/src/onnx/onnx_importer.cpp:275: error: (-210:Unsupported format or combination of formats) Failed to parse ONNX model: /home/dietpi/.deepface/weights/face_recognition_sface_2021dec.onnx in function 'ONNXImporter'` 
The link to weights for SFace needs to be included. Fix:
```sh
mkdir -p ~/.deepface/weights/ && cp face_recognition_sface_2021dec.onnx ~/.deepface/weights/
```
