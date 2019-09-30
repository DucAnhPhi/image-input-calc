# Image-Input Calculator

This program takes images of handwritten equations as an input and outputs the corresponding results.

## Installation

- install Python 3.6 (if not already installed)
- Set up a python virtual environment for this project. It keeps the dependencies/libraries required by different projects in separate places (isolates them to avoid conflicts).

```
$ pip install virtualenv
$ cd image-input-calc
$ virtualenv -p python3.6 virtual_env
```

- to begin using the virtual environment, it needs to be activated:
- Run the following on Linux or MacOS:

```
$ source virtual_env/bin/activate
```

- Run the following if you are working on Windows:

```
$ virtual_env/Scripts/activate.bat
```

- finally install all dependencies running:

```
$ pip install -r requirements.txt
```

- if you are done working in the virtual environment for the moment, you can deactivate it:

```
$ deactivate
```

## Usage
In the project directory, run the program with:

```
$ python3 app.py
```
