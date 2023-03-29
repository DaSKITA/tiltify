# TILTify

## Description


This repository stores the code for TILTify, which is the machine learning component of the Annotation System of Daskita. This component is responsible for providing predictions to the TILTer and learning from the Annotations provided by humans to the TILTer via the Annotation Interface.
The annotations of the policies are performed in the [TILT Schema](https://github.com/Transparency-Information-Language/schema)
The goal is to infer TILT Labels for a given privacy policy and thus perform automated annotations for these policies.


## Project Structure

`/data` - stores data used for experimentation and testing

`/experiments` - stores scripts used for fast experimentation

`/src` - stores the package "tiltify" which shall be installed and used for experimentation

`/test` - Unittests


## SetUp

### Local

This is a setup Guide for Ubuntu.

1. Create a Python Environment with `python -m venv <your-env-name>`
2. Activate your python environment with `source <your-env-name>/bin/activate`
3. Upgrade Pip with `python -m pip install -U pip` and install all necessary packages with `pip install -e src/`
4. Type `source <env-var_file.txt>`. The `<env-var_file.txt>` contains the environment variables listed below.
5. Run the app with `flask run`.


### Environment Variables

Herein you find a list of environment variables needed for the local startup of TILTify.

|Name|Description|
|----|-----------|
|FLASK_APP|Path that points to the `main.py`|
|MONGODB_USERNAME| Username for the TILTer MongoDB|
|MONGODB_PASSWORD| Password for the TILTer MongoDB
|MONGODB_PORT| Port for the TILTer MongoDB|
|MONGO_INITDB_DATABASE| Database for the Annotations and Task in the TILTer MongoDB|
|FLASK_SECRET_KEY| Flask secret key variable|
|JWT_SECRET_KEY| Secret to enable  authorization between TILTer and TILTify|
|TILT_HUB_REST_URL| URL to the TILThub. Has to be requested from Daskita Team|
|TILT_HUB_BASIC_AUTH_USER| TILThub User. Has to be requested from the Daskita Team
|TILT_HUB_BASIC_AUTH_PASSWORD| TILThub password. Has to be requested from the Daskita Team|
|TILTIFY_ADD|TILTify address|
|TILTIFY_PORT| TILTify port|


## License
MIT License

2020

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
