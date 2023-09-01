# Windows OpenSpiel Installation Instructions
Starting with [Windows Subsystem for Linux installation instructions](https://openspiel.readthedocs.io/en/latest/windows.html).
- Run Powershell as administrator.
- `Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux`
- Close Powershell, open Microsoft Store
- Search for Ubuntu, click Get
- Open Ubuntu, create user & password

In WSL:
```bash
sudo apt install python3-pip
python3 -m pip install virtualenv
mkdir envs
cd envs
python3 -m virtualenv dotsboxes
source dotsboxes/bin/activate
cd ..
mkdir repos
cd repos
< clone custom open_spiel branch as specified in project README >
cd open_spiel
./install.sh
pip3 install --upgrade -r requirements.txt
```

Swap to [Installation from Source instructions](https://openspiel.readthedocs.io/en/latest/install.html#installation-from-source):
```bash
./open_spiel/scripts/build_and_run_tests.sh
```

Update PYTHONPATH by adding the following in `~/envs/dotsboxes/bin/activate`:
```bash
# For the python modules in open_spiel.
export PYTHONPATH=$PYTHONPATH:/home/<user>/repos/open_spiel
# For the Python bindings of Pyspiel
export PYTHONPATH=$PYTHONPATH:/home/<user>/repos/open_spiel/build/python
```

If for some reason your `build/` directory is not at `/home/<user>/repos/open_spiel/build/`, update the paths in the snippet above accordingly.

Deactivate and reactivate the virtual environment to update the PYTHONPATH.
```
deactivate
source ~/envs/dotsboxes/bin/activate
```

Make sure the example works:
```
python ~/repos/open_spiel/open_spiel/python/examples/dotsandboxes_example.py
```

Clone the project repository and open in VS Code:
```
cd ~/repos/
git clone https://github.com/alexanderhale/rl-project-KULeuven
cd rl-project-KULeuven
code .
```

Install the project-specific dependencies (those which are not included in OpenSpiel). Be careful when adding dependencies to this file, as they may not be installed on the tournament server.
```
source activate ~/envs/dotsboxes
cd ~/repos/rl-project-KULeuven
pip install -r requirements.txt
```

# SSH Connection Steps
TODO