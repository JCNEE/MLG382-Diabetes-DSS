The point of this is to create virtual environments for your own devices for this project otherwise it's going to install directly on your C Drive:

#1)First Creation Steps#

- Open a terminal in VSCode
- If you dont have python v12: winget install -e --id Python.Python.3.12
- Check if it's installed enter this to verify its installation: py -0p
- py -3.12 -m venv venv


#VERY IMPORTANT:#
To Activate the Virtual Environment and install python libraries run this command

- venv\Scripts\activate

#2) Installing packages for Python#
If you have a requirements.txt (A list of the python libraries you require) run the following command:

- Link the folder your requirements.txt is stored in:
- cd FolderName

If you have one python version installed:
- pip install -r requirements.txt

If you have multiple python versions installed:
- python -m pip install -r requirements.txt
