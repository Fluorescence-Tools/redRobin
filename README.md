# redRobin
redRobin software for CELFIS analysis

<img src="https://user-images.githubusercontent.com/49991393/215458420-9b7ec19c-d7a7-48c2-a329-deadff403a15.jpg" alt="1999roodborst30x30cm" width="350"/>   
Courtesy of Noëlle Koppers - https://noellekoppers.nl/over/

# usage
Checkout the ipython notebook in templateNotebook directory.

# testData
A very limited testdataset is given in testData directory. The size is very limited due to size constraints. For those interested in a larger dataset, please contact the authors of the accompanying scientific publication.

# scientific publication
Manuscript is written and will be published soon.

# author
Software was written by Nicolaas van der Voort.

# Version Interoperability
Written in python3.7 using only libraries available in the standard conda repository.
You may additionally need tiffile and lmfit if you don't have them already:
>pip install tiffile
>pip install lmfit

The relies uses a .dll for reading ptu files and thus it has been tested for windows.
